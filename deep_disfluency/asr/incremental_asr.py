from copy import deepcopy


class ASR(object):
    '''An incremental ASR object which receives new ASR hypotheses
    and maintains the current top level hypothesis.
    '''
    def __init__(self, credentials_file, new_hypothesis_callback,
                 settings=None):
        self.settings = settings
        self.timestamps = True
        if settings and not self.settings.get("timestamps"):
            print "no timestamps"
            self.timestamps = False
        self.credentials_file = credentials_file
        if new_hypothesis_callback:
            self.new_hypothesis_callback = new_hypothesis_callback
        else:
            self.new_hypothesis_callback = None
        self.word_graph = []  # for now, just the top hypothesis

    def get_diff_and_update_word_graph(self, newprefix, verbose=False):
        """Only get the different right frontier according to the timings
        and change the current hypotheses self.word_graph"""
        newprefix = [tuple(p) for p in newprefix]
        if verbose:
            print "word graph", self.word_graph
            print "newprefix", newprefix
        rollback = 0
        original_length = len(self.word_graph)
        original_current = deepcopy(self.word_graph)
        for i in range(len(self.word_graph)-1, -2, -1):
            if verbose:
                print "oooo", newprefix[0]
                if not self.word_graph == []:
                    print self.word_graph[i]
            if (i == -1) or \
                    (float(newprefix[0][1]) >= float(self.word_graph[i][2])):
                if i == len(self.word_graph)-1:
                    self.word_graph = self.word_graph + newprefix
                    break
                k = 0
                marker = i+1
                for j in range(i+1, len(self.word_graph)):
                    if k == len(newprefix):
                        break
                    if verbose:
                        print "...", j, k, self.word_graph[j], newprefix[k],\
                            len(newprefix)
                    if not self.word_graph[j] == newprefix[k]:
                        break
                    else:
                        if verbose:
                            print "repeat"
                        k += 1
                        marker = j+1
                rollback = original_length - marker
                self.word_graph = self.word_graph[:marker] + newprefix[k:]
                newprefix = newprefix[k:]
                break
        if newprefix == []:
            rollback = 0  # just no rollback if no prefix
            self.word_graph = original_current  # reset back to original
        if verbose:
            print "self.word_graph after call", self.word_graph
            print "newprefix after call", newprefix
            print "rollback after call", rollback
        return (newprefix, rollback)

    def listen(self):
        """Simply starts waiting for input and then transcribes."""
        print "listening..."
