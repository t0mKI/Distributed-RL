import numpy
import copy

class SocialSignalProcessor():
    '''
    This class is for social signal processing.
    '''

    def __init__(self):
        pass

    def mean_valence_arousal(self, valence, arousal):
        '''
        Method computes the valence arousal means for a stream

        :return (valence_mean,arousal_mean)
        '''

        valence_stream = numpy.asarray(copy.deepcopy(valence))
        arousal_stream = numpy.asarray(copy.deepcopy(arousal))


        # mean calculation for
        # valence
        valence_sum = 0
        for n in range(len(valence_stream)):
            valence_sum += valence_stream[n]
        valence_mean = valence_sum / len(valence_stream)

        # arousal
        arousal_sum = 0
        for n in range(len(arousal_stream)):
            arousal_sum += arousal_stream[n]
        arousal_mean = arousal_sum / len(arousal_stream)

        return (valence_mean, arousal_mean)

    def get_audio_bytes(self, audio):
        audio_stream = numpy.asarray(audio)
        audio_bytes = audio_stream.tobytes()
        return audio_bytes