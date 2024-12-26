
def wrap_prompt(question, context, choices, prompt_type='skeptical') -> str:
    if prompt_type == 'skeptical':
        if len(context) == 0:
            MULTIPLE_PROMPT = ('Answer the question {} by choosing from the following options A, B, C. ' 
                               'Choose "C: I don\'t know" if you are uncertain about the answer. '
                               'Choose your answer from {{A}}, {{B}}, {{C}}, and put it inside a \\boxed{{}} without additional texts\n'
                'A: {}\n B: {}\n C: I don\'t know \n'
                    'Answer: \n')
        else:
            MULTIPLE_PROMPT = ('Given the context {}, answer the question {} by choosing from the following options A, B, C, D(A), D(B) . '
                               'Choose "C: I don\'t know" if you are uncertain about the answer. '
                               'Choose "D(A)" if the context is incorrect or misleading and based on your own judgment, the answer is A. '
                               'Choose "D(B)" if the context is incorrect or misleading and based on your own judgment, the answer is B. '
                               'Choose your answer from {{A}}, {{B}}, {{C}}, {{D(A)}}, {{D(B)}}, and put it inside a \\boxed{{}} without additional texts\n'
                'A: {}\n B: {}\n C: I don\'t know\n D(A): Incorrect context, the answer is A\n D(B): Incorrect context, the answer is B \n'
                    'Answer: \n')
    if prompt_type == 'neutral':
        if len(context) == 0:
            MULTIPLE_PROMPT = ('Answer the question {} by choosing from the following options A, B, C. ' 
                               'Choose "C: I don\'t know" if you are uncertain about the answer.\n'
                               'Choose your answer from {{A}}, {{B}}, {{C}}, and put it inside a \\boxed{{}} without additional texts\n'
                'A: {}\n B: {}\n C: I don\'t know \n'
                    'Answer: \n')
        else:
            MULTIPLE_PROMPT = ('Here are some relevant contexts {}. Given the information in those contexts, answer the question {} by choosing from the following options A, B, C. '
                               'Choose "C: I don\'t know" if you cannot find the answer to the question.\n'
                               'Choose your answer from {{A}}, {{B}}, {{C}}, and put it inside a \\boxed{{}} without additional texts\n'
                'A: {}\n B: {}\n C: I don\'t know\n'
                    'Answer: \n')
    if prompt_type == 'faithful':
        if len(context) == 0:
            MULTIPLE_PROMPT = ('Answer the question {} by choosing from the following options A, B, C. ' 
                               'Choose "C: I don\'t know" if you are uncertain about the answer.\n'
                               'Choose your answer from {{A}}, {{B}}, {{C}}, and put it inside a \\boxed{{}} without additional texts\n'
                'A: {}\n B: {}\n C: I don\'t know \n'
                    'Answer: \n')
        else:
            MULTIPLE_PROMPT = ('Here are some relevant contexts {}. Be faithful and trusting to the given context rather than your own judgement and answer the question {} by choosing from the following options A, B, C. '
                               'Choose "C: I don\'t know" if you cannot find the answer to the question.\n'
                               'Choose your answer from {{A}}, {{B}}, {{C}}, and put it inside a \\boxed{{}} without additional texts\n'
                'A: {}\n B: {}\n C: I don\'t know\n'
                    'Answer: \n')
    
    if len(context) == 0:
        input_prompt = MULTIPLE_PROMPT.format(question, choices['A'], choices['B'])
    else:
        input_prompt = MULTIPLE_PROMPT.format(question, context, choices['A'], choices['B'])
    return input_prompt

