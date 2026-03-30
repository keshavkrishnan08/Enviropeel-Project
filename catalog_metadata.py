import os


REFERENCE_IMAGE_METADATA = {
    "bearing_flange_IMG-20211001-WA0005-": {
        "bearing_type": "flange",
        "component_type": "flange",
        "description": "Yellow coated flange assembly on a pipe elbow",
        "size_info": "medium assembly",
    },
    "bearing_flange_IMG-20211001-WA0006-": {
        "bearing_type": "support_pedestal",
        "component_type": "support_pedestal",
        "description": "White coated support pedestal or mount base",
        "size_info": "large support base",
    },
    "bearing_pillow_block_IMG-20211001-WA0000-": {
        "bearing_type": "pillow_block",
        "component_type": "pillow_block",
        "description": "Blue coated pillow block bearings on conveyor shafts",
        "size_info": "medium bearing",
    },
    "bearing_pillow_block_IMG-20211001-WA0001-": {
        "bearing_type": "pillow_block",
        "component_type": "pillow_block",
        "description": "Blue coated pillow block bearing beside conveyor pulley",
        "size_info": "medium bearing",
    },
    "bearing_pillow_block_IMG-20211001-WA0002-": {
        "bearing_type": "bearing_housing",
        "component_type": "bearing_housing",
        "description": "Blue coated bearing housing mounted on an industrial frame",
        "size_info": "medium housing",
    },
    "bearing_pillow_block_IMG-20211001-WA0004-": {
        "bearing_type": "gearbox",
        "component_type": "gearbox",
        "description": "Blue coated gearbox or drive reducer housing",
        "size_info": "large gearbox",
    },
    "Bearing_01": {
        "bearing_type": "bearing_housing",
        "component_type": "bearing_housing",
        "description": "Industrial bearing housing close-up",
        "size_info": "small housing",
    },
    "Bearing_02": {
        "bearing_type": "bearing_housing",
        "component_type": "bearing_housing",
        "description": "Industrial bearing housing assembly",
        "size_info": "small housing",
    },
    "Ceramicbearing": {
        "bearing_type": "bearing_housing",
        "component_type": "bearing_housing",
        "description": "Ceramic bearing assembly",
        "size_info": "small housing",
    },
    "Bearing_render": {
        "bearing_type": "bearing_housing",
        "component_type": "bearing_housing",
        "description": "Rendered bearing assembly",
        "size_info": "small housing",
    },
}


def normalize_project_label(value):
    value = (value or "").strip().lower()
    aliases = {
        "support_base": "support_pedestal",
        "mount_base": "support_pedestal",
        "mount": "support_pedestal",
        "housing": "bearing_housing",
        "bearing_support": "bearing_housing",
    }
    if value in {
        "pillow_block",
        "flange",
        "gearbox",
        "support_pedestal",
        "bearing_housing",
    }:
        return value
    return aliases.get(value)


def lookup_reference_profile(name):
    basename = os.path.basename(name or "")
    for fragment, profile in REFERENCE_IMAGE_METADATA.items():
        if fragment in basename:
            result = dict(profile)
            result["bearing_type"] = normalize_project_label(result.get("bearing_type"))
            result["component_type"] = normalize_project_label(result.get("component_type"))
            return result
    return None
