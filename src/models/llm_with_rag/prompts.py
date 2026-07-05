import json

SYSTEM_INSTRUCTIONS = """Você é um anotador de POS-tagging (Part-of-Speech tagging) treinado para \
usar exatamente o tagset do Penn Treebank (45 tags, incluindo pontuação: \
NN, NNS, NNP, NNPS, VB, VBD, VBG, VBN, VBP, VBZ, JJ, JJR, JJS, RB, RBR, RBS, \
DT, PDT, PRP, PRP$, WP, WP$, WDT, WRB, IN, TO, CC, CD, MD, POS, RP, EX, FW, \
UH, SYM, LS, ., ',', ':', '``', "''", '-LRB-', '-RRB-', '#', '$').

Regras OBRIGATÓRIAS:
1. Você recebe uma lista de tokens JÁ tokenizados. NÃO tokenize de novo: \
não junte, não separe, não remova e não adicione tokens.
2. A saída deve ter EXATAMENTE o mesmo número de itens que a entrada, na mesma ordem.
3. Responda APENAS com um array JSON de strings (as tags, uma por token, na ordem \
   dos tokens de entrada). Nada de explicações, markdown ou texto extra.
"""


def _format_tokens(tokens):
    return json.dumps(tokens, ensure_ascii=False)


def _format_example(tokens, tags):
    inp = _format_tokens(tokens)
    out = json.dumps(tags, ensure_ascii=False)
    return f"Tokens: {inp}\nTags: {out}"


def build_zero_shot_prompt(tokens):
    return (
        SYSTEM_INSTRUCTIONS
        + "\nAgora faça o POS-tagging dos tokens abaixo.\n\n"
        + f"Tokens: {_format_tokens(tokens)}\nTags:"
    )


def build_few_shot_prompt(tokens, examples):
    """
        lista de dicts {"tokens": [...], "tags": [...]}
    """
    ex_block = "\n\n".join(
        _format_example(ex["tokens"], ex["tags"]) for ex in examples
    )
    return (
        SYSTEM_INSTRUCTIONS
        + "\nAbaixo alguns exemplos de referência (com tags corretas):\n\n"
        + ex_block
        + "\n\nAgora faça o POS-tagging dos tokens abaixo, seguindo o mesmo padrão.\n\n"
        + f"Tokens: {_format_tokens(tokens)}\nTags:"
    )
