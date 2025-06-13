import os
import numpy as np
import mne
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth

# -----------------------------------------------------------------------------
# 1) Stub: run your FREQ-NESS pipeline on an MNE Raw or Epochs object
# -----------------------------------------------------------------------------
import numpy as np
import mne
from sklearn.decomposition import FastICA


def compute_freq_ness(
    raw, fmin=2.0, fmax=100.0, epoch_length=2.0, n_components=3, filter_design="firwin"
):
    """
    Run a simple FREQ-NESS pipeline on continuous M/EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed (e.g. cleaned, bad channels removed) continuous data.
    fmin, fmax : float
        Min/max frequency (Hz) for CSD estimation and filtering.
    epoch_length : float
        Length in seconds of non-overlapping epochs to chop the data into.
    n_components : int
        Number of independent components (i.e. networks) to extract.
    filter_design : str
        FIR design for band-pass filtering.

    Returns
    -------
    freq_profile : dict
        Mapping from network labels (e.g. 'network_alpha', 'network_beta', ...)
        to a normalized “strength” score in [0,1].
    """
    # 1) Band-pass filter the data to the range of interest
    raw_filt = raw.copy().filter(l_freq=fmin, h_freq=fmax, fir_design=filter_design)

    # 2) Epoch into fixed-length segments
    epochs = mne.make_fixed_length_epochs(
        raw_filt, duration=epoch_length, overlap=0.0, preload=True
    )

    # 3) Compute cross-spectral density matrices via multitaper
    csd = mne.time_frequency.csd_multitaper(
        epochs, fmin=fmin, fmax=fmax, bandwidth=None, n_jobs=1  # pick default bandwidth
    )
    freqs = csd.frequencies  # array of shape (n_freqs,)
    n_freqs = len(freqs)

    # 4) Stack CSD into an array of shape (n_freqs, n_ch, n_ch)
    csd_array = np.stack([csd.get_data(i) for i in range(n_freqs)], axis=0)

    # 5) Vectorize upper-triangle of each CSD matrix → shape (n_freqs, n_pairs)
    n_ch = raw_filt.info["nchan"]
    triu_idx = np.triu_indices(n_ch, k=1)
    X = np.array([csd_array[f][triu_idx] for f in range(n_freqs)])

    # 6) Apply ICA across the frequency axis to find n_components networks
    ica = FastICA(n_components=n_components, random_state=0, max_iter=500)
    S = ica.fit_transform(X)  # shape (n_freqs, n_components)

    # 7) Build a simple frequency profile: for each component, find its peak
    freq_profile = {}
    for comp in range(n_components):
        comp_ts = S[:, comp]
        peak_idx = np.argmax(np.abs(comp_ts))
        peak_freq = freqs[peak_idx]

        # Label according to classical bands
        if 8 <= peak_freq <= 12:
            label = "network_alpha"
        elif 13 <= peak_freq <= 30:
            label = "network_beta"
        elif peak_freq > 30:
            label = "network_gamma"
        else:
            label = f"network_{int(np.round(peak_freq))}Hz"

        # Use absolute amplitude at the peak frequency as the raw score
        freq_profile[label] = float(np.abs(comp_ts[peak_idx]))

    # 8) Normalize so all scores sum to 1
    total = sum(freq_profile.values())
    for key in freq_profile:
        freq_profile[key] /= total

    return freq_profile


# -----------------------------------------------------------------------------
# 2) Map the freq_profile onto Spotify audio‐feature targets
# -----------------------------------------------------------------------------
def map_freq_to_audio_targets(freq_profile):
    """
    Create a dict of audio‐feature “targets” for the Spotify recommendations endpoint.
    """
    # Simple heuristic: stronger alpha → calmer music (lower tempo, lower energy)
    # stronger beta  → more energetic (higher energy, higher danceability)
    # you can refine this as you like!
    af = {}
    af["target_tempo"] = 60 + (1.0 - freq_profile["network_alpha"]) * 120
    af["target_energy"] = freq_profile["network_beta"] * 0.8 + 0.2
    af["target_danceability"] = freq_profile["network_beta"] * 0.7 + 0.3
    af["target_valence"] = freq_profile["network_gamma"] * 0.5 + 0.5
    return af


# -----------------------------------------------------------------------------
# 3) Authenticate to Spotify and build the playlist
# -----------------------------------------------------------------------------
def create_brain_synced_playlist(raw, playlist_name="Brain-Sync Mix"):
    # --- Compute FREQ-NESS profile ---
    freq_profile = compute_freq_ness(raw)
    audio_targets = map_freq_to_audio_targets(freq_profile)

    # --- Spotify OAuth setup ---
    sp = Spotify(
        auth_manager=SpotifyOAuth(
            scope="playlist-modify-private",
            client_id=os.environ["SPOTIPY_CLIENT_ID"],
            client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
            redirect_uri="http://localhost:8888/callback",
        )
    )

    user_id = sp.current_user()["id"]

    # --- Get recommendations ---
    rec = sp.recommendations(
        seed_genres=["ambient", "chill"],  # you can choose seeds dynamically
        limit=30,
        **audio_targets,
    )

    track_uris = [t["uri"] for t in rec["tracks"]]

    # --- Create a new private playlist and add tracks ---
    playlist = sp.user_playlist_create(
        user_id,
        name=playlist_name,
        public=False,
        description="Auto‐generated to match your brain’s FREQ-NESS profile",
    )
    sp.playlist_add_items(playlist["id"], track_uris)

    print(f"Created playlist '{playlist_name}' with {len(track_uris)} tracks:")
    for t in rec["tracks"]:
        print(f"  • {t['name']} — {t['artists'][0]['name']}")


# -----------------------------------------------------------------------------
# 4) Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) load your raw MEG/EEG (MNE) object:
    # raw = mne.io.read_raw_fif("path/to/data_raw.fif", preload=True)
    #
    # For this demo we'll fake it:
    raw = None

    create_brain_synced_playlist(raw)


def analyze_spotify_tracks(sp, track_uris):
    """
    Get audio features for a list of Spotify tracks.

    Parameters
    ----------
    sp : spotipy.Spotify
        Authenticated Spotify client
    track_uris : list
        List of Spotify track URIs

    Returns
    -------
    track_features : list of dict
        Audio features for each track
    """
    # Spotify API returns max 100 tracks at once
    features = []
    for i in range(0, len(track_uris), 100):
        batch = track_uris[i : i + 100]
        batch_features = sp.audio_features(batch)
        features.extend(batch_features)

    return features


def score_track_compatibility(freq_profile, track_features):
    """
    Score how well a track's audio features match the brain's frequency profile.

    Parameters
    ----------
    freq_profile : dict
        Output from compute_freq_ness()
    track_features : dict
        Spotify audio features for a single track

    Returns
    -------
    compatibility_score : float
        Score between 0-1 indicating how well the track matches brain state
    """
    if not track_features:  # Handle None responses from Spotify API
        return 0.0

    # Convert brain state to expected audio features
    brain_targets = map_freq_to_audio_targets(freq_profile)

    # Calculate similarity scores for key features
    scores = []

    # Tempo matching
    tempo_diff = abs(track_features["tempo"] - brain_targets["target_tempo"])
    tempo_score = max(0, 1 - (tempo_diff / 100))  # Normalize tempo difference
    scores.append(tempo_score)

    # Energy matching
    energy_diff = abs(track_features["energy"] - brain_targets["target_energy"])
    scores.append(1 - energy_diff)

    # Danceability matching
    dance_diff = abs(
        track_features["danceability"] - brain_targets["target_danceability"]
    )
    scores.append(1 - dance_diff)

    # Valence matching
    valence_diff = abs(track_features["valence"] - brain_targets["target_valence"])
    scores.append(1 - valence_diff)

    # Return weighted average
    return np.mean(scores)


def filter_existing_playlist(raw, playlist_id, min_compatibility=0.6):
    """
    Filter an existing Spotify playlist through FREQ-NESS analysis.

    Parameters
    ----------
    raw : mne.io.Raw
        Brain data for computing frequency profile
    playlist_id : str
        Spotify playlist ID to filter
    min_compatibility : float
        Minimum compatibility score to keep tracks

    Returns
    -------
    filtered_tracks : list
        Tracks that match the brain state above threshold
    """
    # Compute brain frequency profile
    freq_profile = compute_freq_ness(raw)

    # Setup Spotify client
    sp = Spotify(
        auth_manager=SpotifyOAuth(
            scope="playlist-read-private",
            client_id=os.environ["SPOTIPY_CLIENT_ID"],
            client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
            redirect_uri="http://localhost:8888/callback",
        )
    )

    # Get playlist tracks
    playlist_tracks = sp.playlist_tracks(playlist_id)
    track_uris = [item["track"]["uri"] for item in playlist_tracks["items"]]

    # Analyze audio features
    track_features = analyze_spotify_tracks(sp, track_uris)

    # Score and filter tracks
    compatible_tracks = []
    for i, features in enumerate(track_features):
        score = score_track_compatibility(freq_profile, features)
        if score >= min_compatibility:
            track_info = playlist_tracks["items"][i]["track"]
            compatible_tracks.append(
                {
                    "track": track_info,
                    "compatibility_score": score,
                    "uri": track_uris[i],
                }
            )

    # Sort by compatibility score (best matches first)
    compatible_tracks.sort(key=lambda x: x["compatibility_score"], reverse=True)

    return compatible_tracks


def filter_multiple_sources(raw, source_playlists=None, recommendation_seeds=None):
    """
    Filter tracks from multiple sources (playlists, recommendations, etc.)

    Parameters
    ----------
    raw : mne.io.Raw
        Brain data
    source_playlists : list
        List of playlist IDs to filter
    recommendation_seeds : dict
        Seeds for getting fresh recommendations to filter

    Returns
    -------
    all_compatible_tracks : list
        All compatible tracks from all sources, ranked by compatibility
    """
    freq_profile = compute_freq_ness(raw)
    all_tracks = []

    sp = Spotify(
        auth_manager=SpotifyOAuth(
            scope="playlist-read-private",
            client_id=os.environ["SPOTIPY_CLIENT_ID"],
            client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
            redirect_uri="http://localhost:8888/callback",
        )
    )

    # Process existing playlists
    if source_playlists:
        for playlist_id in source_playlists:
            compatible = filter_existing_playlist(raw, playlist_id)
            all_tracks.extend(compatible)

    # Process fresh recommendations
    if recommendation_seeds:
        audio_targets = map_freq_to_audio_targets(freq_profile)
        recs = sp.recommendations(limit=50, **recommendation_seeds, **audio_targets)

        track_uris = [t["uri"] for t in recs["tracks"]]
        features = analyze_spotify_tracks(sp, track_uris)

        for i, track_features in enumerate(features):
            score = score_track_compatibility(freq_profile, track_features)
            if score >= 0.6:  # threshold
                all_tracks.append(
                    {
                        "track": recs["tracks"][i],
                        "compatibility_score": score,
                        "uri": track_uris[i],
                    }
                )

    # Remove duplicates and sort
    seen_uris = set()
    unique_tracks = []
    for track in all_tracks:
        if track["uri"] not in seen_uris:
            unique_tracks.append(track)
            seen_uris.add(track["uri"])

    unique_tracks.sort(key=lambda x: x["compatibility_score"], reverse=True)
    return unique_tracks


def create_filtered_brain_playlist(
    raw, source_playlist_ids, new_playlist_name="Filtered Brain-Sync"
):
    """
    Create a new playlist from filtered existing content
    """
    # Filter existing playlists through brain analysis
    compatible_tracks = filter_multiple_sources(
        raw,
        source_playlists=source_playlist_ids,
        recommendation_seeds={"seed_genres": ["ambient", "electronic", "chill"]},
    )

    # Take top 30 most compatible tracks
    top_tracks = compatible_tracks[:30]
    track_uris = [t["uri"] for t in top_tracks]

    # Create new playlist
    sp = Spotify(
        auth_manager=SpotifyOAuth(
            scope="playlist-modify-private",
            client_id=os.environ["SPOTIPY_CLIENT_ID"],
            client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
            redirect_uri="http://localhost:8888/callback",
        )
    )

    user_id = sp.current_user()["id"]
    playlist = sp.user_playlist_create(user_id, new_playlist_name, public=False)
    sp.playlist_add_items(playlist["id"], track_uris)

    print(f"Created filtered playlist with {len(track_uris)} brain-compatible tracks")
    for track in top_tracks[:10]:  # Show top 10
        print(
            f"  • {track['track']['name']} (compatibility: {track['compatibility_score']:.2f})"
        )
