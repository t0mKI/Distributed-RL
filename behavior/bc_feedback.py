
class BCFeedback:

    XML_TAG = 'BCFeedback'
    def __init__(self, bc_style):
        self.bc_style=bc_style


# reactive expression-> reactive expression-> short, lexical phrase/word
class ReactiveExprLexicalWord(BCFeedback):
    XML_TAG = 'ReactiveExprLexicalWord'
    SINGLE_EXPRESSIONS=['okay', 'alright' ,'right', 'I see']
    # reactive expressions (partially overlapping with acknowledgement:
    # “oh really/really”, “yeah”, “gee”, “okay”, “sure”, “exactly”, “allright”, “man”, “shit”, “hell”

    def __init__(self, bc_style, lexical_bc_expr):
        BCFeedback.__init__(self, bc_style)
        self.lexical_bc_expr=lexical_bc_expr


# reactive expression-> backchannel->nonlexical audio expressions
class BackchannelAudio(BCFeedback):
    XML_TAG = 'BackchannelAudio'
    SINGLE_EXPRESSIONS=[ "<spurt audio=\"" + 'g0001_014' + "\">a</spurt> ",
                         "ah <prosody speed=\"fast\"><emphasis>huh</emphasis></prosody>"


    ]

    # bc audio (partially overlapping with acknowledgement:
    # “hm”, “huh”, “oh”,
    #     # “mhm”, “uh huh”

    def __init__(self, bc_style, lexical_bc_expr):
        BCFeedback.__init__(self,bc_style)
        self.lexical_bc_expr=lexical_bc_expr


# reactive expression-> acknowledgement
class Acknowledgement(BCFeedback):
    XML_TAG = 'Acknowledgement'
    #SINGLE_EXPRESSIONS=['']
    SINGLE_EXPRESSIONS = ['yeah', 'yes']

    # ‘‘Yeah’’, ‘‘Mm hm’’

    def __init__(self, bc_style, lexical_bc_expr):
        BCFeedback.__init__(self, bc_style)
        self.lexical_bc_expr=lexical_bc_expr



class ResumptiveOpener(BCFeedback):
    XML_TAG = 'ResumptiveOpener'
    ssml_gesture='g0001_037'

    def __init__(self, bc_style, lexical_bc_expr):
        BCFeedback.__init__(self, bc_style)
        self.lexical_bc_expr = lexical_bc_expr

# repetition + uh huh/ok/i see/yes
class Repetition(BCFeedback):
    XML_TAG = 'Repetition'
    SINGLE_EXPRESSIONS=['railway']

    def __init__(self, bc_style, lexical_bc_expr):
        BCFeedback.__init__(self,bc_style)
        self.lexical_bc_expr = lexical_bc_expr



