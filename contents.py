METHOD = "METHOD"
DOT = "DOT"
LPAREN = "LPAREN"
VARIABLE = "VARIABLE"
TYPE = "TYPE"
METHOD_SELECT = "METHOD_SELECT"
MEMBER_SELECT = "MEMBER_SELECT"
EXPRESSION = "EXPRESSION"
STATEMENTS = "STATEMENTS"
METHOD_INVOCATION = "METHOD_INVOCATION"
LBRACE = "LBRACE"
NEW = "NEW"
BLOCK = "BLOCK"
BODY = "BODY"
PARAM_TYPE = "PARAMETERIZED_TYPE"
MODIFIERS = "MODIFIERS"
RETURN_TYPE = "RETURN_TYPE"
THIS = "this"
COMPILATION_UNIT = "COMPILATION_UNIT"
STRING_LITERAL = "STRING_LITERAL"
CHAR_LITERAL = "CHAR_LITERAL"
ANNOTATIONS = "ANNOTATIONS"

translate_dict = {
    "UNDERSCORE": "_",
    "ARROW": "->",
    "COLCOL": "::",
    "LPAREN": "(",
    "RPAREN": ")",
    "LBRACE": "{",
    "RBRACE": "}",
    "LBRACKET": "[",
    "RBRACKET": "]",
    "SEMI": ";",
    "COMMA": ",",
    "DOT": ".",
    "ELLIPSIS": "...",
    "EQ": "=",
    "GT": ">",
    "LT": "<",
    "BANG": "!",
    "TILDE": "~",
    "QUES": "?",
    "COLON": ":",
    "EQEQ": "==",
    "LTEQ": "<=",
    "GTEQ": ">=",
    "BANGEQ": "!=",
    "AMPAMP": "&&",
    "BARBAR": "||",
    "PLUSPLUS": "++",
    "SUBSUB": "--",
    "PLUS": "+",
    "SUB": "-",
    "STAR": "*",
    "SLASH": "/",
    "AMP": "&",
    "BAR": "|",
    "CARET": "^",
    "PERCENT": "%",
    "LTLT": "<<",
    "GTGT": ">>",
    "GTGTGT": ">>>",
    "PLUSEQ": "+=",
    "SUBEQ": "-=",
    "STAREQ": "*=",
    "SLASHEQ": "/=",
    "AMPEQ": "&=",
    "BAREQ": "|=",
    "CARETEQ": "^=",
    "PERCENTEQ": "%=",
    "LTLTEQ": "<<=",
    "GTGTEQ": ">>=",
    "GTGTGTEQ": ">>>=",
    "MONKEYS_AT": "@"
}