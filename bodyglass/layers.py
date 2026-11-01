"""Plain-language, educational descriptions. Not medical advice."""

LAYERS = [
    {
        "id": "skin", "name": "Skin & scalp", "color": "#e8b89a", "opacity": 0.35,
        "material": "skin", "approx": False,
        "description": "The outer surface of the head: skin, the fat beneath it and the scalp. "
                       "In MRI it is the boundary between the head and the surrounding air.",
    },
    {
        "id": "skull", "name": "Skull (estimated)", "color": "#efe6d2", "opacity": 0.25,
        "material": "bone", "approx": True,
        "description": "The bony case that protects the brain. Bone gives almost no MRI signal, "
                       "so this shell is estimated from the space between the brain and scalp; "
                       "it is a rough illustration, not the true bone shape.",
    },
    {
        "id": "brain", "name": "Brain surface (cortex)", "color": "#eaa6a2", "opacity": 1.0,
        "material": "brain", "approx": True,
        "description": "The cerebral cortex, the folded outer layer of grey matter. Its ridges "
                       "(gyri) and grooves (sulci) pack a large surface into the skull. The cortex "
                       "handles perception, movement, language, memory and thinking. "
                       "Also shown: the cerebellum at the back (balance and coordination) and the brainstem.",
    },
    {
        "id": "white_matter", "name": "White matter", "color": "#f3efe6", "opacity": 1.0,
        "material": "wm", "approx": True,
        "description": "Bundles of insulated nerve fibres (axons wrapped in myelin) that connect "
                       "brain regions to each other, like the brain's wiring. Myelin is fatty, which "
                       "is why white matter looks bright on a T1 scan.",
    },
    {
        "id": "ventricles", "name": "Lateral ventricles", "color": "#5fb6e8", "opacity": 1.0,
        "material": "fluid", "approx": True,
        "description": "Two C-shaped fluid-filled spaces deep in the brain. They make and hold "
                       "cerebrospinal fluid (CSF), which cushions the brain and carries away waste. "
                       "Their size varies a lot between healthy people.",
    },
]
