import math

def glm5_vision_canvas(
    num_frames,
    height,
    width,
    temporal_factor,
    factor,
    min_tokens,
    max_tokens
):
    """
    Aligned canvas within a spatiotemporal TOKEN budget
    (min/max given in tokens of temporal_factor * factor^2 pixels each)
    """

    pixels_per_token = temporal_factor * factor ** 2
    min_pixels = min_tokens * pixels_per_token
    max_pixels = max_tokens * pixels_per_token

    def align(value, f):
        return math.ceil(value / f) * f

    aligned_frames = max(temporal_factor, round(num_frames / temporal_factor) * temporal_factor)
    aligned_height = align(height, factor)
    aligned_width = align(width, factor)
    budget = aligned_frames * aligned_height * aligned_width

    if budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = align(max(1, math.ceil(height * scale)), factor)
        aligned_width = align(max(1, math.ceil(width * scale)), factor)
        budget = aligned_frames * aligned_height * aligned_width

    if budget > max_pixels:
        low, high = 1, height
        best_h = best_w = factor
        while low <= high:
            ch = (low + high) // 2
            cw = max(1, math.floor(width * ch / height))
            cand_h, cand_w = align(ch, factor), align(cw, factor)
            if aligned_frames * cand_h * cand_w <= max_pixels:
                best_h, best_w = cand_h, cand_w
                low = ch + 1
            else:
                high = ch - 1
        aligned_height, aligned_width = best_h, best_w

    return aligned_height, aligned_width
