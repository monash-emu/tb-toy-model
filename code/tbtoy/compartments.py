"""
Compartment definitions for the simplified TB model.

The natural history structure is identical to the full model: individuals move
across the TB infection / disease spectrum, with disease states differentiated by
symptom status (subclinical vs clinical) and infectiousness (low vs high).
"""

COMPARTMENTS = (
    "mtb_naive",  # never infected
    # Early TB infection states
    "incipient",
    "contained",
    "cleared",
    # TB disease states
    "subclin_lowinf",
    "clin_lowinf",
    "subclin_inf",
    "clin_inf",
    # Treatment and recovery
    "treatment",
    "recovered",
)

LATENT_COMPS = ["incipient", "contained", "cleared"]
ACTIVE_COMPS = ["subclin_lowinf", "clin_lowinf", "subclin_inf", "clin_inf"]
INFECTIOUS_COMPS = ACTIVE_COMPS

# States for which a viable infection is still present (i.e. can still progress to disease)
VIABLE_INFECTION_COMPS = ["incipient", "contained"]

INFECTION_SOURCE_COMPS = ["mtb_naive", "contained", "cleared", "recovered"]
