"""Associative conditioning: three regions of one array play US, CS and NS.

After Gallinaro, Gašparović & Rotter (2022, *PLoS Comput Biol*, Fig 1), which is a simulation of a
network with homeostatic structural plasticity. Here the same design is run on a culture: an
unconditioned stimulus **US**, a conditioned stimulus **CS** always paired with it, and a control
**NS** stimulated on its own. Every region is probed alone before, after, and after rests. The
question is whether probing CS comes to evoke activity in US that probing NS does not, how long
that lasts, and whether it keeps growing after the pairing stops (as structural plasticity
predicts) or decays at once (as STDP would).

Open loop: the whole run is a schedule decided in advance (:mod:`.protocol`), executed by one
script (:mod:`.run`) that fires each presentation on time, and read out afterwards from the
recording (:mod:`.report`). Choosing where the three regions go is :mod:`.select`. Nothing here
needs ``maxlab`` except :mod:`.run` and the rig steps of :mod:`.select`, which import it on use.

Two short runs gate the long one, and both report back before a culture is committed
(:mod:`.checks`): ``calibration`` finds each region's amplitude, and ``connectivity`` measures how
much stimulating each region alone drives the other two. Sites that turn out not to be
independent, or not connected at all, are worth knowing about before four hours of conditioning.

    python -m mxtreme.experiments.associative preview  --params params.json
    python -m mxtreme.experiments.associative select   --params params.json --scan-npz ... --out ...
    python -m mxtreme.experiments.associative run      --params <the file select wrote> --mode calibration
    python -m mxtreme.experiments.associative run      --params <same, amplitudes filled in> --mode connectivity
    python -m mxtreme.experiments.associative run      --params <same> --config mxtreme.toml
    python -m mxtreme.experiments.associative report   <recording>.raw.h5 --protocol <run dir>/..._protocol.json
"""
