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
(:mod:`.checks`): ``calibration`` finds each region's amplitude, and the session's baseline block measures how
much stimulating each region alone drives the other two. Sites that turn out not to be
independent, or not connected at all, are worth knowing about before four hours of conditioning.

    # after braintrix-cli's activity scan and network scan (the baseline) of the culture:
    python -m mxtreme.experiments.associative select   --params params_default.json --out <work dir> \
        --baseline <..._network_scan.raw.h5> --activity-scan <..._activity_scan.raw.h5> --set batch=... ...
    python -m mxtreme.experiments.associative preview  --params <the file select wrote> --config mxtreme.toml
    python -m mxtreme.experiments.associative run      --params <same> --config mxtreme.toml --mode calibration
    python -m mxtreme.experiments.associative compare  <calibration>.raw.h5 <calibration, other pattern>.raw.h5
    python -m mxtreme.experiments.associative report   <calibration>.raw.h5 --apply <the parameter file>
    python -m mxtreme.experiments.associative run      --params <same, amplitudes filled in> --config mxtreme.toml --phase session
    python -m mxtreme.experiments.associative report   <every recording of the session>.raw.h5 -o <work dir>/session.png

The store gets the recordings and nothing else; each carries its own protocol, so ``report`` needs
only the files. A conditioning session is one command but one recording per phase -- baseline, each encode cycle,
retrieval -- each closed before the next opens, so each can be read as soon as it is done, and all
of them together as one at the end. ``select``'s parameter file and figures, and every report, go in a work directory
outside it. docs/concepts/associative-experiment-day.md is the full procedure.
"""
