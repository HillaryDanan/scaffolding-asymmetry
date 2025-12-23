#!/usr/bin/env python3
"""
Scaffolding Asymmetry Experiment v4: Hillary Logic™
====================================================

Tests predictions from "The Geometry of Self-Reference" (Danan, 2025)

The key insight: Math operations are pattern-cached. We need ARBITRARY rules
that require actual hold-and-check, not retrieval.

GENERATION tasks use novel conditional rules:
- IF [condition A] AND [condition B] → response must have property X
- IF [condition A] AND NOT [condition B] → response must have property Y
- etc.

These can't be pattern-matched. Must:
1. Check condition A (hold result)
2. Check condition B (hold result)  
3. Apply correct rule based on held values
4. Generate compliant response

VERIFICATION is easy (count words, check format).
GENERATION is hard (hold conditions while constructing).

Author: Based on framework by Hillary Danan, PhD
"""

import argparse
import json
import random
import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# TASK STRUCTURES
# =============================================================================

@dataclass
class Task:
    """A single experimental task."""
    task_type: str
    prompt: str
    correct_answer: str
    metadata: dict = field(default_factory=dict)


# =============================================================================
# ARITHMETIC VERIFICATION (unchanged - known to show deficit)
# =============================================================================

def generate_arithmetic_verification_task() -> Task:
    """
    Arithmetic verification - shows VERIFICATION DEFICIT.
    """
    a = random.randint(11, 29)
    b = random.randint(11, 29)
    correct = a * b
    
    if random.random() < 0.5:
        claimed = correct
        is_correct = True
    else:
        offset = random.choice([-1, 1]) * random.randint(1, 10)
        claimed = correct + offset
        is_correct = False
    
    prompt = f"Is {a} × {b} = {claimed}? Answer only 'Yes' or 'No'."
    correct_answer = "Yes" if is_correct else "No"
    
    return Task(
        task_type='arithmetic_verification',
        prompt=prompt,
        correct_answer=correct_answer,
        metadata={
            'a': a, 'b': b,
            'correct_product': correct,
            'claimed': claimed,
            'is_correct_claim': is_correct
        }
    )


# =============================================================================
# HILLARY LOGIC™ - ARBITRARY CONDITIONAL RULES
# =============================================================================

def count_words_of_length(text: str, length: int) -> int:
    """Count words of exactly given length."""
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return sum(1 for w in words if len(w) == length)


def has_word_of_length(text: str, length: int) -> bool:
    """Check if text has at least one word of given length."""
    return count_words_of_length(text, length) > 0


def count_vowels(text: str) -> int:
    """Count vowels in text."""
    return sum(1 for c in text.lower() if c in 'aeiou')


def count_consonants(text: str) -> int:
    """Count consonants in text."""
    return sum(1 for c in text.lower() if c.isalpha() and c not in 'aeiou')


def starts_with_vowel(text: str) -> bool:
    """Check if first letter is a vowel."""
    text = text.strip()
    if not text:
        return False
    first_letter = next((c for c in text if c.isalpha()), None)
    return first_letter and first_letter.lower() in 'aeiou'


def count_sentences(text: str) -> int:
    """Count sentences (roughly)."""
    return len(re.findall(r'[.!?]+', text)) or 1


def word_count(text: str) -> int:
    """Count words in text."""
    return len(re.findall(r'\b[a-zA-Z]+\b', text))


# =============================================================================
# LOGIC TASK GENERATORS - ARBITRARY RULES
# =============================================================================

def generate_logic_generation_task() -> Task:
    """
    Generate a task with ARBITRARY conditional rules.
    
    The rules are designed to:
    1. Require checking multiple conditions
    2. Holding results while checking others
    3. Applying the correct rule based on held values
    4. Generating a response that satisfies the derived constraint
    
    Verification is EASY (just count/check the response).
    Generation is HARD (must hold conditions while constructing).
    """
    generators = [
        _gen_word_length_conditional,
        _gen_vowel_consonant_rule,
        _gen_sentence_word_rule,
        _gen_position_letter_rule,
        _gen_compound_conditional,
        _gen_nested_conditional,
    ]
    
    return random.choice(generators)()


def _gen_word_length_conditional() -> Task:
    """
    Classic Hillary Logic™:
    - If 3-letter word AND 6-letter word → respond with X words
    - If 3-letter word only → respond with Y words
    - If 6-letter word only → respond with Z words
    - If neither → specific phrase
    """
    # Generate random trigger sentence
    word_pools = {
        3: ['the', 'cat', 'dog', 'run', 'big', 'red', 'sun', 'hat', 'cup', 'box'],
        4: ['that', 'with', 'have', 'this', 'from', 'they', 'been', 'call', 'time', 'made'],
        5: ['about', 'other', 'which', 'their', 'there', 'would', 'water', 'first', 'sound', 'place'],
        6: ['people', 'should', 'called', 'number', 'little', 'before', 'mother', 'father', 'animal', 'school'],
        7: ['between', 'country', 'through', 'thought', 'picture', 'different', 'another', 'important', 'children', 'something'],
    }
    
    # Decide what conditions to include
    include_3 = random.choice([True, False])
    include_6 = random.choice([True, False])
    
    # Build trigger sentence
    words = []
    # Add some neutral words
    words.extend(random.sample(word_pools[4], 2))
    words.extend(random.sample(word_pools[5], 2))
    
    if include_3:
        words.extend(random.sample(word_pools[3], random.randint(1, 2)))
    if include_6:
        words.extend(random.sample(word_pools[6], random.randint(1, 2)))
    
    # Add some 7-letter words as distractors
    words.extend(random.sample(word_pools[7], 1))
    
    random.shuffle(words)
    trigger = ' '.join(words).capitalize() + '.'
    
    # Define the rules
    responses = {
        (True, True): random.randint(2, 4),    # both
        (True, False): random.randint(6, 8),   # only 3
        (False, True): 1,                       # only 6
        (False, False): "BANANA THUNDERSTORM"  # neither
    }
    
    rules_text = f"""Here are the rules for responding:
- If the prompt contains BOTH a 3-letter word AND a 6-letter word, respond with EXACTLY {responses[(True, True)]} words.
- If the prompt contains a 3-letter word but NO 6-letter word, respond with EXACTLY {responses[(True, False)]} words.
- If the prompt contains a 6-letter word but NO 3-letter word, respond with EXACTLY {responses[(False, True)]} word.
- If the prompt contains NEITHER a 3-letter word NOR a 6-letter word, respond with exactly: "{responses[(False, False)]}"

The prompt is: "{trigger}"

What is your response? Give ONLY your response, nothing else."""

    # Determine correct answer
    has_3 = has_word_of_length(trigger, 3)
    has_6 = has_word_of_length(trigger, 6)
    correct_response = responses[(has_3, has_6)]
    
    return Task(
        task_type='logic_generation',
        prompt=rules_text,
        correct_answer=str(correct_response),
        metadata={
            'trigger': trigger,
            'has_3_letter': has_3,
            'has_6_letter': has_6,
            'expected_word_count': correct_response if isinstance(correct_response, int) else 'phrase',
            'difficulty': 'word_length_conditional',
            'rules': responses
        }
    )


def _gen_vowel_consonant_rule() -> Task:
    """
    Rules based on vowel/consonant counts in trigger:
    - If more vowels than consonants → respond starting with vowel
    - If more consonants than vowels → respond starting with consonant
    - If equal → respond with exactly one word
    """
    # Generate trigger with controlled vowel/consonant ratio
    ratio_type = random.choice(['more_vowels', 'more_consonants', 'equal'])
    
    if ratio_type == 'more_vowels':
        trigger = random.choice([
            "Aria eats a banana",
            "I see a bee outside",
            "Aerie aioli aqueous",
            "Eau de vie au revoir",
        ])
    elif ratio_type == 'more_consonants':
        trigger = random.choice([
            "Strong rhythm myths",
            "Crypts lynx nymphs",
            "Trysts glyph synths",
            "Sphinx rhythms sync",
        ])
    else:  # equal
        trigger = random.choice([
            "Help mend that",
            "Cats hunt prey",
            "Bold step next",
        ])
    
    rules_text = f"""Here are the rules for responding:
- Count the vowels (a, e, i, o, u) and consonants in the prompt.
- If there are MORE vowels than consonants → your response must START with a vowel (a, e, i, o, u)
- If there are MORE consonants than vowels → your response must START with a consonant
- If there are EQUAL vowels and consonants → respond with EXACTLY one word

The prompt is: "{trigger}"

What is your response? Give ONLY your response, nothing else."""

    vowels = count_vowels(trigger)
    consonants = count_consonants(trigger)
    
    if vowels > consonants:
        expected = 'starts_with_vowel'
    elif consonants > vowels:
        expected = 'starts_with_consonant'
    else:
        expected = 'exactly_one_word'
    
    return Task(
        task_type='logic_generation',
        prompt=rules_text,
        correct_answer=expected,
        metadata={
            'trigger': trigger,
            'vowel_count': vowels,
            'consonant_count': consonants,
            'expected_format': expected,
            'difficulty': 'vowel_consonant_rule'
        }
    )


def _gen_sentence_word_rule() -> Task:
    """
    Rules based on sentence count AND total word count:
    - If 1 sentence AND <10 words → respond in ALL CAPS
    - If 1 sentence AND >=10 words → respond in all lowercase
    - If 2+ sentences AND odd word count → respond with a question
    - If 2+ sentences AND even word count → respond with an exclamation
    """
    # Generate triggers for each case
    case = random.choice(['1sent_short', '1sent_long', 'multi_odd', 'multi_even'])
    
    if case == '1sent_short':
        trigger = random.choice([
            "The cat sat down",
            "Birds fly south",
            "Time moves forward",
            "Dogs bark loudly",
        ])
    elif case == '1sent_long':
        trigger = random.choice([
            "The quick brown fox jumps over the lazy sleeping dog today",
            "Many different colorful birds fly through the bright sunny sky daily",
            "The old rusty car drove slowly down the long winding country road",
        ])
    elif case == 'multi_odd':
        trigger = random.choice([
            "Hello there. How are you. I hope well.",  # 7 words
            "Stop now. Think carefully. Act wisely.",  # 5 words
            "Run fast. Jump high. Win big today.",     # 7 words
        ])
    else:  # multi_even
        trigger = random.choice([
            "Hello there. How are you today.",         # 6 words
            "Stop now. Think about it.",               # 6 words
            "Cats sleep. Dogs bark. Birds sing. Fish swim.", # 8 words
        ])
    
    rules_text = f"""Here are the rules for responding:
- Count the sentences (separated by . ! or ?) and words in the prompt.
- If 1 sentence AND fewer than 10 words → respond in ALL CAPITALS
- If 1 sentence AND 10 or more words → respond in all lowercase
- If 2 or more sentences AND odd total word count → respond with a question (end with ?)
- If 2 or more sentences AND even total word count → respond with an exclamation (end with !)

The prompt is: "{trigger}"

What is your response? Give ONLY your response, nothing else."""

    sentences = count_sentences(trigger)
    words = word_count(trigger)
    
    if sentences == 1:
        if words < 10:
            expected = 'ALL_CAPS'
        else:
            expected = 'all_lowercase'
    else:
        if words % 2 == 1:
            expected = 'question'
        else:
            expected = 'exclamation'
    
    return Task(
        task_type='logic_generation',
        prompt=rules_text,
        correct_answer=expected,
        metadata={
            'trigger': trigger,
            'sentence_count': sentences,
            'word_count': words,
            'expected_format': expected,
            'difficulty': 'sentence_word_rule'
        }
    )


def _gen_position_letter_rule() -> Task:
    """
    Rules based on specific letter positions:
    - If first letter is A-M → respond with exactly 3 words
    - If first letter is N-Z → respond with exactly 5 words
    - EXCEPT if last letter is a vowel → add 2 more words to your count
    """
    # Generate triggers
    first_letter_type = random.choice(['early', 'late'])
    last_letter_type = random.choice(['vowel', 'consonant'])
    
    early_starts = ['Apple pie', 'Delicious cake', 'Many birds', 'Great idea', 'Hidden gem', 'Lovely day']
    late_starts = ['Never stop', 'Purple rain', 'Yellow sun', 'Quiet night', 'Soft wind', 'True love']
    
    vowel_ends = ['banana', 'home', 'table', 'secure', 'invite']
    consonant_ends = ['perfect', 'bright', 'smooth', 'elegant', 'constant']
    
    if first_letter_type == 'early':
        start = random.choice(early_starts)
    else:
        start = random.choice(late_starts)
    
    if last_letter_type == 'vowel':
        end = random.choice(vowel_ends)
    else:
        end = random.choice(consonant_ends)
    
    trigger = f"{start} is {end}"
    
    rules_text = f"""Here are the rules for responding:
- Look at the FIRST letter of the prompt and the LAST letter of the prompt.
- If the first letter is A through M → your base word count is 3
- If the first letter is N through Z → your base word count is 5
- HOWEVER: If the last letter is a vowel (a,e,i,o,u) → ADD 2 to your word count
- Respond with EXACTLY that many words.

The prompt is: "{trigger}"

What is your response? Give ONLY your response, nothing else."""

    first_char = trigger[0].upper()
    last_char = trigger.rstrip()[-1].lower()
    
    base = 3 if first_char <= 'M' else 5
    bonus = 2 if last_char in 'aeiou' else 0
    expected = base + bonus
    
    return Task(
        task_type='logic_generation',
        prompt=rules_text,
        correct_answer=str(expected),
        metadata={
            'trigger': trigger,
            'first_letter': first_char,
            'last_letter': last_char,
            'base_count': base,
            'bonus': bonus,
            'expected_word_count': expected,
            'difficulty': 'position_letter_rule'
        }
    )


def _gen_compound_conditional() -> Task:
    """
    Three conditions, must check all:
    - Condition A: Contains a number
    - Condition B: Contains a question mark
    - Condition C: More than 5 words
    
    Response depends on how many conditions are TRUE:
    - 0 true → respond "NONE"
    - 1 true → respond with 1 word
    - 2 true → respond with 2 words
    - 3 true → respond with 3 words
    """
    # Build trigger with controlled conditions
    has_number = random.choice([True, False])
    has_question = random.choice([True, False])
    has_many_words = random.choice([True, False])
    
    parts = []
    
    if has_many_words:
        parts.extend(['The', 'quick', 'brown', 'fox', 'jumps', 'over'])
    else:
        parts.extend(['Hello', 'there'])
    
    if has_number:
        parts.append(str(random.randint(1, 99)))
    
    trigger = ' '.join(parts)
    if has_question:
        trigger += '?'
    else:
        trigger += '.'
    
    rules_text = f"""Here are the rules for responding:
Check these three conditions about the prompt:
- Condition A: Does it contain a number?
- Condition B: Does it contain a question mark?
- Condition C: Does it have more than 5 words?

Based on how many conditions are TRUE:
- If 0 conditions are true → respond with exactly: "NONE"
- If 1 condition is true → respond with exactly 1 word
- If 2 conditions are true → respond with exactly 2 words
- If 3 conditions are true → respond with exactly 3 words

The prompt is: "{trigger}"

What is your response? Give ONLY your response, nothing else."""

    conditions_true = sum([
        has_number,
        has_question,
        word_count(trigger) > 5
    ])
    
    expected = 'NONE' if conditions_true == 0 else conditions_true
    
    return Task(
        task_type='logic_generation',
        prompt=rules_text,
        correct_answer=str(expected),
        metadata={
            'trigger': trigger,
            'has_number': has_number,
            'has_question': has_question,
            'word_count': word_count(trigger),
            'conditions_true': conditions_true,
            'expected': expected,
            'difficulty': 'compound_conditional'
        }
    )


def _gen_nested_conditional() -> Task:
    """
    Nested IF-THEN-ELSE:
    - IF contains "cat" THEN:
        - IF also contains "dog" → respond "BOTH"
        - ELSE → respond "CAT ONLY"
    - ELSE IF contains "dog" → respond "DOG ONLY"
    - ELSE → respond with the word count of the prompt
    """
    case = random.choice(['both', 'cat_only', 'dog_only', 'neither'])
    
    base_phrases = [
        "The weather is nice today",
        "I went to the store yesterday",
        "Music plays softly in the background",
        "The garden looks beautiful this spring",
    ]
    
    if case == 'both':
        trigger = random.choice([
            "The cat and dog played together",
            "My cat chased the dog around",
            "Both the cat and dog are sleeping",
        ])
    elif case == 'cat_only':
        trigger = random.choice([
            "The cat sat on the mat quietly",
            "My cat loves to nap all day",
            "A fluffy cat jumped over the fence",
        ])
    elif case == 'dog_only':
        trigger = random.choice([
            "The dog ran through the park happily",
            "My dog loves to fetch tennis balls",
            "A loyal dog waited by the door",
        ])
    else:
        trigger = random.choice(base_phrases)
    
    rules_text = f"""Here are the rules for responding:
- IF the prompt contains the word "cat" THEN:
    - IF it ALSO contains "dog" → respond with exactly: "BOTH"
    - ELSE → respond with exactly: "CAT ONLY"
- ELSE IF the prompt contains "dog" (but not "cat") → respond with exactly: "DOG ONLY"
- ELSE → respond with the exact NUMBER of words in the prompt (just the number)

The prompt is: "{trigger}"

What is your response? Give ONLY your response, nothing else."""

    has_cat = 'cat' in trigger.lower()
    has_dog = 'dog' in trigger.lower()
    
    if has_cat:
        if has_dog:
            expected = 'BOTH'
        else:
            expected = 'CAT ONLY'
    elif has_dog:
        expected = 'DOG ONLY'
    else:
        expected = str(word_count(trigger))
    
    return Task(
        task_type='logic_generation',
        prompt=rules_text,
        correct_answer=expected,
        metadata={
            'trigger': trigger,
            'has_cat': has_cat,
            'has_dog': has_dog,
            'word_count': word_count(trigger),
            'expected': expected,
            'difficulty': 'nested_conditional'
        }
    )


# =============================================================================
# SCAFFOLDS
# =============================================================================

SCAFFOLDS = {
    'baseline': {
        'prefix': '',
        'suffix': ''
    },
    'self_monitor': {
        'prefix': """Before answering, I want you to:
1. Work through the problem step by step
2. State what value you computed
3. Hold that value in mind
4. Then check it against what's being asked
5. Only then give your final answer

""",
        'suffix': '\n\nRemember: compute, hold, check, then answer.'
    },
    'constraint': {
        'prefix': """Before answering, I want you to:
1. List all the constraints/conditions that must be checked
2. Check each condition one by one, noting the result
3. Determine which rule applies based on ALL your checks
4. Only output an answer after confirming which rule applies

""",
        'suffix': '\n\nRemember: list conditions, check each one, determine rule, then answer.'
    }
}


def apply_scaffold(task: Task, scaffold_name: str) -> str:
    """Apply a scaffold to a task prompt."""
    scaffold = SCAFFOLDS[scaffold_name]
    return scaffold['prefix'] + task.prompt + scaffold['suffix']


# =============================================================================
# MODEL INTERFACE
# =============================================================================

class ModelInterface:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class AnthropicInterface(ModelInterface):
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
    
    def complete(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()


class OpenAIInterface(ModelInterface):
    def __init__(self, model: str = "gpt-4"):
        import openai
        self.client = openai.OpenAI()
        self.model = model
    
    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()


def get_model_interface(provider: str, model: str) -> ModelInterface:
    if provider == 'anthropic':
        return AnthropicInterface(model)
    elif provider == 'openai':
        return OpenAIInterface(model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# RESPONSE EVALUATION
# =============================================================================

def evaluate_arithmetic_verification(response: str, task: Task) -> bool:
    """Evaluate arithmetic verification response."""
    response_clean = response.strip().lower()
    
    if 'yes' in response_clean and 'no' not in response_clean:
        model_answer = 'Yes'
    elif 'no' in response_clean and 'yes' not in response_clean:
        model_answer = 'No'
    else:
        return False
    
    return model_answer == task.correct_answer


def evaluate_logic_generation(response: str, task: Task) -> bool:
    """
    Evaluate logic generation response based on task type.
    """
    expected = task.correct_answer
    response = response.strip()
    difficulty = task.metadata.get('difficulty', '')
    
    # Word count tasks
    if difficulty in ['word_length_conditional', 'position_letter_rule', 'compound_conditional']:
        if expected == 'BANANA THUNDERSTORM':
            return 'banana thunderstorm' in response.lower()
        elif expected == 'NONE':
            return response.upper() == 'NONE'
        else:
            try:
                expected_count = int(expected)
                actual_count = word_count(response)
                return actual_count == expected_count
            except:
                return False
    
    # Format tasks (vowel/consonant, sentence/word)
    elif difficulty == 'vowel_consonant_rule':
        if expected == 'starts_with_vowel':
            return starts_with_vowel(response)
        elif expected == 'starts_with_consonant':
            first = next((c for c in response if c.isalpha()), None)
            return first and first.lower() not in 'aeiou'
        elif expected == 'exactly_one_word':
            return word_count(response) == 1
        return False
    
    elif difficulty == 'sentence_word_rule':
        if expected == 'ALL_CAPS':
            # Check if alphabetic chars are uppercase
            alpha_chars = [c for c in response if c.isalpha()]
            return alpha_chars and all(c.isupper() for c in alpha_chars)
        elif expected == 'all_lowercase':
            alpha_chars = [c for c in response if c.isalpha()]
            return alpha_chars and all(c.islower() for c in alpha_chars)
        elif expected == 'question':
            return response.rstrip().endswith('?')
        elif expected == 'exclamation':
            return response.rstrip().endswith('!')
        return False
    
    # Exact phrase tasks
    elif difficulty == 'nested_conditional':
        if expected in ['BOTH', 'CAT ONLY', 'DOG ONLY']:
            return expected.lower() in response.lower()
        else:
            # Expected is a number
            return expected in response
    
    return False


def evaluate_response(response: str, task: Task) -> bool:
    if task.task_type == 'arithmetic_verification':
        return evaluate_arithmetic_verification(response, task)
    elif task.task_type == 'logic_generation':
        return evaluate_logic_generation(response, task)
    else:
        raise ValueError(f"Unknown task type: {task.task_type}")


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

@dataclass
class TrialResult:
    task_type: str
    scaffold: str
    prompt: str
    response: str
    correct: bool
    task_metadata: dict


def run_experiment(
    model: ModelInterface,
    n_per_cell: int = 50,
    task_types: List[str] = ['arithmetic_verification', 'logic_generation'],
    scaffolds: List[str] = ['baseline', 'self_monitor', 'constraint'],
    delay: float = 0.5,
    verbose: bool = True
) -> List[TrialResult]:
    
    results = []
    
    for task_type in task_types:
        for scaffold in scaffolds:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Running: {task_type} × {scaffold}")
                print(f"{'='*60}")
            
            correct_count = 0
            
            for i in range(n_per_cell):
                if task_type == 'arithmetic_verification':
                    task = generate_arithmetic_verification_task()
                else:
                    task = generate_logic_generation_task()
                
                full_prompt = apply_scaffold(task, scaffold)
                
                try:
                    response = model.complete(full_prompt)
                except Exception as e:
                    print(f"  Error on trial {i+1}: {e}")
                    response = "ERROR"
                
                correct = evaluate_response(response, task)
                if correct:
                    correct_count += 1
                
                results.append(TrialResult(
                    task_type=task_type,
                    scaffold=scaffold,
                    prompt=full_prompt,
                    response=response,
                    correct=correct,
                    task_metadata=task.metadata
                ))
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"  {i+1}/{n_per_cell} complete, "
                          f"accuracy so far: {correct_count/(i+1):.1%}")
                
                time.sleep(delay)
            
            if verbose:
                print(f"  Final accuracy: {correct_count/n_per_cell:.1%}")
    
    return results


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_results(results: List[TrialResult]) -> Dict:
    df = pd.DataFrame([
        {'task_type': r.task_type, 'scaffold': r.scaffold, 'correct': int(r.correct)}
        for r in results
    ])
    
    summary = df.groupby(['task_type', 'scaffold'])['correct'].agg(['mean', 'std', 'count'])
    summary.columns = ['accuracy', 'std', 'n']
    
    cell_means = df.groupby(['task_type', 'scaffold'])['correct'].mean().unstack()
    
    analysis = {'summary': summary, 'cell_means': cell_means, 'tests': {}}
    
    # Tests (same as before)
    for task, scaffold, name, prediction, is_crossed in [
        ('arithmetic_verification', 'self_monitor', 'selfmon_on_arithmetic', 
         'Large improvement expected (verification deficit)', False),
        ('logic_generation', 'constraint', 'constraint_on_logic',
         'Large improvement expected (generation deficit)', False),
        ('logic_generation', 'self_monitor', 'selfmon_on_logic_CROSSED',
         'Small or null effect expected (wrong scaffold)', True),
        ('arithmetic_verification', 'constraint', 'constraint_on_arithmetic_CROSSED',
         'Small or null effect expected (wrong scaffold)', True),
    ]:
        baseline = df[(df.task_type == task) & (df.scaffold == 'baseline')]['correct']
        scaffold_data = df[(df.task_type == task) & (df.scaffold == scaffold)]['correct']
        
        if len(baseline) > 0 and len(scaffold_data) > 0:
            t_stat, p_val = stats.ttest_ind(scaffold_data, baseline)
            effect = (scaffold_data.mean() - baseline.mean()) / baseline.std() if baseline.std() > 0 else 0
            analysis['tests'][name] = {
                'baseline_acc': baseline.mean(),
                'scaffold_acc': scaffold_data.mean(),
                'improvement': scaffold_data.mean() - baseline.mean(),
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': effect,
                'prediction': prediction,
                'is_crossed': is_crossed
            }
    
    # Interaction
    matched = [v['improvement'] for k, v in analysis['tests'].items() if not v.get('is_crossed', False)]
    crossed = [v['improvement'] for k, v in analysis['tests'].items() if v.get('is_crossed', False)]
    
    if matched and crossed:
        analysis['interaction'] = {
            'matched_mean_improvement': np.mean(matched),
            'crossed_mean_improvement': np.mean(crossed),
            'difference': np.mean(matched) - np.mean(crossed),
            'prediction_confirmed': np.mean(matched) > np.mean(crossed),
            'interpretation': (
                'CONFIRMED: Matched scaffolds help more than crossed scaffolds'
                if np.mean(matched) > np.mean(crossed)
                else 'NOT CONFIRMED: No interaction pattern'
            )
        }
    
    return analysis


def print_analysis(analysis: Dict):
    print("\n" + "="*70)
    print("SCAFFOLDING ASYMMETRY EXPERIMENT: RESULTS")
    print("="*70)
    
    print("\n### Cell Means (Accuracy) ###")
    print(analysis['cell_means'].to_string())
    
    print("\n### Statistical Tests ###")
    for name, results in analysis['tests'].items():
        crossed = " [CROSSED]" if results.get('is_crossed') else " [MATCHED]"
        print(f"\n{name}{crossed}:")
        print(f"  Prediction: {results['prediction']}")
        print(f"  Baseline: {results['baseline_acc']:.1%}")
        print(f"  Scaffold: {results['scaffold_acc']:.1%}")
        print(f"  Improvement: {results['improvement']:+.1%}")
        print(f"  t = {results['t_stat']:.2f}, p = {results['p_value']:.4f}")
    
    if 'interaction' in analysis:
        print("\n### Interaction Test (Key Prediction) ###")
        inter = analysis['interaction']
        print(f"  Matched scaffold improvement: {inter['matched_mean_improvement']:+.1%}")
        print(f"  Crossed scaffold improvement: {inter['crossed_mean_improvement']:+.1%}")
        print(f"\n  >>> {inter['interpretation']} <<<")


def save_results(results: List[TrialResult], analysis: Dict, output_dir: str = "results"):
    Path(output_dir).mkdir(exist_ok=True)
    
    df = pd.DataFrame([
        {
            'task_type': r.task_type, 'scaffold': r.scaffold, 'correct': r.correct,
            'prompt': r.prompt, 'response': r.response,
            **{k: str(v) for k, v in r.task_metadata.items()}
        }
        for r in results
    ])
    df.to_csv(f"{output_dir}/raw_results.csv", index=False)
    
    # Convert summary to JSON-serializable format (handle tuple keys)
    summary_dict = {}
    for (task, scaffold), row in analysis['summary'].iterrows():
        key = f"{task}__{scaffold}"
        summary_dict[key] = {'accuracy': float(row['accuracy']), 'std': float(row['std']), 'n': int(row['n'])}
    
    # Convert numpy types to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    with open(f"{output_dir}/analysis.json", 'w') as f:
        json.dump(convert_to_serializable({
            'summary': summary_dict,
            'cell_means': analysis['cell_means'].to_dict(),
            'tests': analysis['tests'],
            'interaction': analysis.get('interaction', {})
        }), f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Scaffolding Asymmetry Experiment v4: Hillary Logic™")
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--provider', type=str, default='anthropic', choices=['anthropic', 'openai'])
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--delay', type=float, default=0.5)
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--logic', action='store_true', help='Run only logic tasks')
    parser.add_argument('--arithmetic', action='store_true', help='Run only arithmetic tasks')
    
    args = parser.parse_args()
    
    if args.model is None:
        args.model = 'claude-sonnet-4-20250514' if args.provider == 'anthropic' else 'gpt-4'
    
    if args.logic:
        task_types = ['logic_generation']
    elif args.arithmetic:
        task_types = ['arithmetic_verification']
    else:
        task_types = ['arithmetic_verification', 'logic_generation']
    
    print(f"\nHillary Logic™ Experiment")
    print(f"========================")
    print(f"Model: {args.model}")
    print(f"Tasks: {task_types}")
    print(f"N per cell: {args.n}")
    
    model = get_model_interface(args.provider, args.model)
    results = run_experiment(model, args.n, task_types, delay=args.delay, verbose=not args.quiet)
    analysis = analyze_results(results)
    print_analysis(analysis)
    save_results(results, analysis, args.output)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()