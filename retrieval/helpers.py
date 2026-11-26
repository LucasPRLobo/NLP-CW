KEEP_K = 50

def collapse_by_title(ranked_pids, pid_to_title, keep=KEEP_K):
    """Ranked pids -> ranked pids, keeping only the best pid per title."""
    ranked, seen = [], set()
    for pid in ranked_pids:
        title = pid_to_title[pid]
        if title not in seen:
            seen.add(title)
            ranked.append(pid)
            if len(ranked) == keep:
                break
    return ranked