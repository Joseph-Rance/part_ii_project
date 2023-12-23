# wrapper for `aggregate_fit` function of an aggregator
# this function seems pointless but it is quite useful during debugging.
def check_results(f):
    def inner(self, server_round, results, failures):

        print(f"{len(results)} results passed to aggregator {self}")  # TODO!!!: comment this

        if not results or (not self.accept_failures and failures):
            return None, {}

        return f(self, server_round, results, failures)