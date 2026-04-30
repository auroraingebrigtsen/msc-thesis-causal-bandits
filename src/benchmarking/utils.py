def compute_change_points(T:int, change_params) -> list[int]:
    """Helper function to get change points based on the specified change mode and parameters.
        TODO: Add support for more change modes like "random" etc.
    """
    if change_params.get("mode") == "controlled":
        every = change_params.get("every", 0)
        if every <= 0:
            return []
        return list(range(every, T, every))