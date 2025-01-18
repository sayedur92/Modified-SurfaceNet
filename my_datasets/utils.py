# Add sketch to textures_mapping with 1 channel
textures_mapping = {
    "diffuse": 3,
    "normal": 3,
    "roughness": 1,
    "specular": 3,
}


# Update texture_maps to include sketch
texture_maps = list(textures_mapping.keys())




# Validation function
def validate_textures(textures):
    """
    Validate if the requested textures are in the predefined textures_mapping.

    Args:
        textures (list): List of requested texture maps.

    Raises:
        Exception: If a requested map is not found in textures_mapping.
    """
    invalid_maps = [x for x in textures if x not in textures_mapping.keys()]
    if invalid_maps:
        raise Exception(
            f"Requested maps must be in: {list(textures_mapping.keys())}. "
            f"Invalid maps: {invalid_maps}"
        )
