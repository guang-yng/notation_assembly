import logging
import os

from mung.grammar import DependencyGrammar
from mung.io import parse_node_classes

def load_grammar(filename):
    mungo_classes_file = os.path.splitext(filename)[0] + '.xml'
    mlclass_dict = {m.name: m for m in parse_node_classes(mungo_classes_file)}
    g = DependencyGrammar(grammar_filename=filename, alphabet=set(mlclass_dict.keys()))
    return g

def config2data_pool_dict(config):
    """Prepare data pool kwargs from an exp_config dict.

    Grammar file is loaded relative to the munglinker/ package."""
    data_pool_dict = {
        'max_edge_length': config['THRESHOLD_NEGATIVE_DISTANCE'],
        'max_negative_samples': config['MAX_NEGATIVE_EXAMPLES_PER_OBJECT'],
        'patch_size': (config['PATCH_HEIGHT'], config['PATCH_WIDTH']),
        'zoom': config['IMAGE_ZOOM']
    }

    if 'PATCH_NO_IMAGE' in config:
        data_pool_dict['patch_no_image'] = config['PATCH_NO_IMAGE']

    # Load grammar, if requested
    if 'RESTRICT_TO_GRAMMAR' in config:
        if not os.path.isfile(config['RESTRICT_TO_GRAMMAR']):
            grammar_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        config['RESTRICT_TO_GRAMMAR'])
        else:
            grammar_path = config['RESTRICT_TO_GRAMMAR']

        if os.path.isfile(grammar_path):
            grammar = load_grammar(grammar_path)
            data_pool_dict['grammar'] = grammar
        else:
            logging.warning('Config contains grammar {}, but it is unreachable'
                            ' both as an absolute path and relative to'
                            ' the munglinker/ package. No grammar loaded.'
                            ''.format(config['RESTRICT_TO_GRAMMAR']))

    return data_pool_dict