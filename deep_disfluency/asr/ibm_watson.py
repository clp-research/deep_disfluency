import itertools

import fluteline
import watson_streaming


class IBMWatsonAdapter(fluteline.Consumer):
    '''
    A fluteline consumer-producer that receives transcription from
    :class:`watson_streaming.Transcriber` and prepare them to work
    with the deep_disfluency tagger.
    '''
    def enter(self):
        self.running_id = itertools.count()
        # Messages that went down the pipeline, indexed by start_time.
        self.memory = {}
        # Track when Watson commits changes to clear the memory.
        self.result_index = 0

    def consume(self, data):
        if 'results' in data:
            self.clear_memory_if_necessary(data)
            for t in data['results'][0]['alternatives'][0]['timestamps']:
                self.process_timestamp(t)

    def clear_memory_if_necessary(self, data):
        if data['result_index'] > self.result_index:
            self.memory = {}
            self.result_index = data['result_index']

    def process_timestamp(self, timestamp):
        word, start_time, end_time = timestamp
        word = self.clean_word(word)

        if self.is_new(start_time):
            id_ = next(self.running_id)
        elif self.is_update(start_time, word):
            id_ = self.memory[start_time]['id']
        else:
            id_ = None

        if id_ is not None:
            msg = {
                'start_time': start_time,
                'end_time': end_time,
                'word': word,
                'id': id_
            }
            self.memory[start_time] = msg
            self.put(msg)


    def clean_word(self, word):
        if word in ['mmhm', 'aha', 'uhhuh']:
            return 'uh-huh'
        if word == '%HESITATION':
            return 'uh'
        return word

    def is_new(self, start_time):
        return start_time not in self.memory

    def is_update(self, start_time, word):
        return self.memory[start_time]['word'] != word
