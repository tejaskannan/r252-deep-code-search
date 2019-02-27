
# AST node contents used when parsing java files
METHOD = 'METHOD'
DOT = 'DOT'
LPAREN = 'LPAREN'
VARIABLE = 'VARIABLE'
TYPE = 'TYPE'
METHOD_SELECT = 'METHOD_SELECT'
MEMBER_SELECT = 'MEMBER_SELECT'
EXPRESSION = 'EXPRESSION'
STATEMENTS = 'STATEMENTS'
METHOD_INVOCATION = 'METHOD_INVOCATION'
LBRACE = 'LBRACE'
NEW = 'NEW'
BLOCK = 'BLOCK'
BODY = 'BODY'
PARAM_TYPE = 'PARAMETERIZED_TYPE'
MODIFIERS = 'MODIFIERS'
RETURN_TYPE = 'RETURN_TYPE'
THIS = 'this'
NEW_LOWER = 'new'
COMPILATION_UNIT = 'COMPILATION_UNIT'
STRING_LITERAL = 'STRING_LITERAL'
CHAR_LITERAL = 'CHAR_LITERAL'
ANNOTATIONS = 'ANNOTATIONS'

# Line used for formatting
LINE = '-' * 50

# File names for model saving
META_NAME = 'meta.pkl.gz'
MODEL_NAME = 'model.chk'

BIG_NUMBER = 1e7
SMALL_NUMBER = 1e-7

# Data file names
JAVADOC_FILE_NAME = 'javadoc.txt'
METHOD_NAME_FILE_NAME = 'method-names.txt'
METHOD_API_FILE_NAME = 'method-apis.txt'
METHOD_TOKENS_FILE_NAME = 'method-tokens.txt'

# Output formats used when parsing
API_FORMAT = '{0}.{1}'
TOKEN_FORMAT = '{0} '
STRING_FORMAT = '\"{0}\"'
CHAR_FORMAT = '\'{0}\''

# Formats for training outputs
DATE_FORMAT = '%Y-%m-%d-%H-%M-%S'
NAME_FORMAT = '{0}-{1}-{2}'
LOG_FORMAT = '{0}{1}-data.csv'

translate_dict = {
    'UNDERSCORE': '_',
    'ARROW': '->',
    'COLCOL': '::',
    'LPAREN': '(',
    'RPAREN': ')',
    'LBRACE': '{',
    'RBRACE': '}',
    'LBRACKET': '[',
    'RBRACKET': ']',
    'SEMI': ';',
    'COMMA': ',',
    'DOT': '.',
    'ELLIPSIS': '...',
    'EQ': '=',
    'GT': '>',
    'LT': '<',
    'BANG': '!',
    'TILDE': '~',
    'QUES': '?',
    'COLON': ':',
    'EQEQ': '==',
    'LTEQ': '<=',
    'GTEQ': '>=',
    'BANGEQ': '!=',
    'AMPAMP': '&&',
    'BARBAR': '||',
    'PLUSPLUS': '++',
    'SUBSUB': '--',
    'PLUS': '+',
    'SUB': '-',
    'STAR': '*',
    'SLASH': '/',
    'AMP': '&',
    'BAR': '|',
    'CARET': '^',
    'PERCENT': '%',
    'LTLT': '<<',
    'GTGT': '>>',
    'GTGTGT': '>>>',
    'PLUSEQ': '+=',
    'SUBEQ': '-=',
    'STAREQ': '*=',
    'SLASHEQ': '/=',
    'AMPEQ': '&=',
    'BAREQ': '|=',
    'CARETEQ': '^=',
    'PERCENTEQ': '%=',
    'LTLTEQ': '<<=',
    'GTGTEQ': '>>=',
    'GTGTGTEQ': '>>>=',
    'MONKEYS_AT': '@'
}