"""Useful helper functions."""

#from logging import INFO
#from flwr.common.logger import log

def check_results(f):
    """Wrapper for the `aggregate_fit` function of an aggregator, convenient for debugging."""

    def inner(self, server_round, results, failures):

        #log(INFO, f"{len(results)} results passed to aggregator {self}")

        if not results or (not self.accept_failures and failures):
            return None, {}

        return f(self, server_round, results, failures)
    return inner
