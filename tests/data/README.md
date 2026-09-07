# Snapshot fixtures

Real chains pulled from Massive land here, one JSON per `(ticker, as_of)`:

    python -m examples.spx_surface --fetch      # needs POLYGON_API_KEY

They are committed on purpose. A quote is a fact about one instant, so anything
derived from a live pull is unreproducible by construction, and a test that
re-fetches fails for reasons unrelated to the code.

This directory is empty of real snapshots because none has been pulled yet. The
test suite and the example both run on a manufactured chain from
`torch_pricer.data.synthetic` instead — which is not a stopgap: a generated chain
has a known right answer, so the pipeline can be asked to *recover* a forward and
a smile rather than merely to produce plausible ones. A real snapshot can only
ever be checked for self-consistency.
