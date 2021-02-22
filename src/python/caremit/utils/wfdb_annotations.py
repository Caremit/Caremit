from collections import namedtuple

annotation_fields = "sample symbol subtype chan num aux_note".split(" ")
record_fields = "record_name fs label_store description custom_labels contained_labels".split(" ")
Annotation = namedtuple("Annotation", annotation_fields + record_fields)


class AnnotationCollection:
    def __init__(self, wfdb_annotation_object):
        self.ann = wfdb_annotation_object

    def __getitem__(self, i):
        d = {}
        for field in annotation_fields:
            d[field] = self.ann.__dict__[field][i]
        for field in record_fields:
            d[field] = self.ann.__dict__[field]

        return Annotation(**d)

    def __iter__(self):
        return (self[i] for i in range(self.ann.ann_len))


symbol_dict = {
    ' ': 'Not an actual annotation',
    'N': 'Normal beat',
    'L': 'Left bundle branch block beat',
    'R': 'Right bundle branch block beat',
    'a': 'Aberrated atrial premature beat',
    'V': 'Premature ventricular contraction',
    'F': 'Fusion of ventricular and normal beat',
    'J': 'Nodal (junctional) premature beat',
    'A': 'Atrial premature contraction',
    'S': 'Premature or ectopic supraventricular beat',
    'E': 'Ventricular escape beat',
    'j': 'Nodal (junctional) escape beat',
    '/': 'Paced beat',
    'Q': 'Unclassifiable beat',
    '~': 'Signal quality change',
    '|': 'Isolated QRS-like artifact',
    's': 'ST change',
    'T': 'T-wave change',
    '*': 'Systole',
    'D': 'Diastole',
    '"': 'Comment annotation',
    '=': 'Measurement annotation',
    'p': 'P-wave peak',
    'B': 'Left or right bundle branch block',
    '^': 'Non-conducted pacer spike',
    't': 'T-wave peak',
    '+': 'Rhythm change',
    'u': 'U-wave peak',
    '?': 'Learning',
    '!': 'Ventricular flutter wave',
    '[': 'Start of ventricular flutter/fibrillation',
    ']': 'End of ventricular flutter/fibrillation',
    'e': 'Atrial escape beat',
    'n': 'Supraventricular escape beat',
    '@': 'Link to external data (aux_note contains URL)',
    'x': 'Non-conducted P-wave (blocked APB)',
    'f': 'Fusion of paced and normal beat',
    '(': 'Waveform onset',
    ')': 'Waveform end',
    'r': 'R-on-T premature ventricular contraction'
}