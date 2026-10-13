"""Lexicon-based clinical relevance filter (explored, not part of the GhostCluster framework).

The manuscript reports that this post-processing filter gave inconsistent results across corpora
(Section 6.5). It is provided so that this statement can be reproduced.
A Ghost Cluster passes when at least 30% of its distinct expressions contain two or more word
tokens and at least one token from the health lexicon below (361 terms).
"""
import re

BODY_PARTS = {
    'head', 'brain', 'eye', 'eyes', 'ear', 'ears', 'nose', 'mouth',
    'tongue', 'tooth', 'teeth', 'throat', 'neck', 'chest', 'heart',
    'lung', 'lungs', 'stomach', 'gut', 'guts', 'liver', 'kidney',
    'bowel', 'intestine', 'bladder', 'skin', 'bone', 'bones',
    'muscle', 'muscles', 'nerve', 'nerves', 'joint', 'joints',
    'back', 'spine', 'arm', 'arms', 'leg', 'legs', 'hand', 'hands',
    'foot', 'feet', 'finger', 'fingers', 'toe', 'toes', 'face',
    'jaw', 'lip', 'lips', 'hip', 'hips', 'shoulder', 'shoulders',
    'knee', 'knees', 'ankle', 'ankles', 'wrist', 'wrists',
    'belly', 'tummy', 'rib', 'ribs', 'vein', 'veins',
    'blood', 'body', 'hair', 'nails', 'breast', 'prostate',
    'uterus', 'ovary', 'thyroid', 'endocrine', 'nervous',
}

SYMPTOMS = {
    'pain', 'ache', 'aches', 'hurt', 'hurts', 'hurting', 'sore',
    'burn', 'burns', 'burning', 'itch', 'itchy', 'itching',
    'rash', 'hives', 'bleed', 'bleeding', 'swell', 'swelling',
    'swollen', 'numb', 'numbness', 'tingle', 'tingling',
    'dizzy', 'dizziness', 'vertigo', 'lightheaded',
    'nausea', 'nauseous', 'vomit', 'vomiting', 'puke', 'puking',
    'headache', 'migraine', 'fatigue', 'fatigued', 'exhausted',
    'tired', 'tiredness', 'weak', 'weakness', 'weaker',
    'fever', 'chills', 'shiver', 'shivering', 'sweating', 'sweaty',
    'cough', 'coughing', 'sneeze', 'sneezing', 'wheeze', 'wheezing',
    'cramp', 'cramps', 'cramping', 'crampy', 'spasm', 'spasms',
    'seizure', 'seizures', 'tremor', 'tremors', 'shaking', 'shaky',
    'stiff', 'stiffness', 'rigid', 'inflammation', 'inflamed',
    'infection', 'infected', 'allergy', 'allergic',
    'diarrhea', 'diarrhoea', 'constipation', 'constipated',
    'bloating', 'bloated', 'gas', 'gassy', 'indigestion',
    'heartburn', 'reflux', 'flatulence',
    'blurry', 'blurred', 'vision', 'hearing', 'deaf', 'tinnitus',
    'breathing', 'breathless', 'breath', 'palpitation', 'palpitations',
    'faint', 'fainting', 'fainted', 'faintness', 'blackout',
    'insomnia', 'restless', 'restlessness', 'drowsy', 'drowsiness',
    'sedated', 'sedation', 'comatose', 'lethargic', 'lethargy',
}

PSYCH_TERMS = {
    'anxiety', 'anxious', 'panic', 'panicky', 'phobia',
    'depression', 'depressed', 'depressive', 'suicidal', 'suicide',
    'hallucination', 'hallucinations', 'hallucinating', 'delusion',
    'paranoia', 'paranoid', 'psychosis', 'psychotic', 'mania', 'manic',
    'mood', 'moody', 'irritable', 'irritability', 'agitation', 'agitated',
    'anger', 'angry', 'rage', 'crying', 'cried', 'cries', 'tearful',
    'emotional', 'emotions', 'anhedonia', 'apathy', 'apathetic',
    'dissociation', 'dissociating', 'derealisation', 'depersonalisation',
    'obsessive', 'compulsive', 'intrusive', 'rumination', 'ruminating',
    'confused', 'confusion', 'disoriented', 'cognitive', 'concentration',
    'memory', 'forgetful', 'forgetfulness', 'brainfog',
}

DRUG_RELATED = {
    'withdrawal', 'withdrawals', 'discontinuation', 'taper', 'tapering',
    'dose', 'dosage', 'overdose', 'medication', 'medicine', 'drug',
    'pill', 'pills', 'tablet', 'capsule', 'prescription', 'rx',
    'side', 'effect', 'effects', 'reaction', 'adverse', 'symptom',
    'symptoms', 'syndrome', 'tolerance', 'dependence', 'dependent',
    'addiction', 'addicted', 'addictive', 'rebound',
}

EXPERIENTIAL = {
    'sleep', 'sleeping', 'sleepy', 'sleepless', 'asleep', 'awake',
    'appetite', 'hungry', 'hunger', 'cravings', 'craving', 'binge',
    'weight', 'gained', 'gaining', 'losing', 'pounds', 'lbs',
    'libido', 'sexual', 'orgasm', 'erection', 'ejaculation', 'impotence',
    'energy', 'wired', 'jittery', 'restless',
    'eat', 'eating', 'swallow', 'swallowing',
    'urinate', 'urination', 'pee', 'peeing', 'urine',
}

INFORMAL_HEALTH = {
    'zap', 'zaps', 'zapping', 'buzz', 'buzzing',
    'fog', 'foggy', 'haze', 'hazy', 'cloudy',
    'zombie', 'zombified', 'zonked', 'zoned',
    'woozy', 'wobbly', 'spacey', 'spaced',
    'puffy', 'bloat', 'groggy',
    'achy', 'throbbing', 'stabbing', 'shooting',
    'twitching', 'twitch', 'twitchy', 'tics', 'tic',
    'grinding', 'clenching',
    'flushing', 'flushed', 'flush',
    'poop', 'pooping', 'diarrea',
    'throw', 'threw', 'thrown',  # "threw up"
    'numb', 'dead', 'flat',  # emotional states
}

# Combined lexicon
HEALTH_LEXICON = (
    BODY_PARTS | SYMPTOMS | PSYCH_TERMS | DRUG_RELATED |
    EXPERIENTIAL | INFORMAL_HEALTH
)


def tokens(text):
    return re.findall(r"[a-z]+", str(text).lower())


def expression_passes(text, min_tokens=2):
    t = tokens(text)
    return len(t) >= min_tokens and any(w in HEALTH_LEXICON for w in t)


def cluster_passes(expressions, min_share=0.3):
    expressions = list(dict.fromkeys(expressions))
    return sum(expression_passes(e) for e in expressions) / max(len(expressions), 1) >= min_share
