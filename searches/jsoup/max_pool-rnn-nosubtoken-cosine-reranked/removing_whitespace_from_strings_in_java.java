public class Results {

  private static void appendWhitespaceIfBr(Element element, StringBuilder accum) {
    if (element.tag.getName().equals("br") && !TextNode.lastCharIsWhitespace(accum))
      accum.append(" ");
  }

  public static void appendNormalisedWhitespace(
      StringBuilder accum, String string, boolean stripLeading) {
    boolean lastWasWhite = false;
    boolean reachedNonWhite = false;
    int len = string.length();
    int c;
    for (int i = 0; i < len; i += Character.charCount(c)) {
      c = string.codePointAt(i);
      if (isActuallyWhitespace(c)) {
        if ((stripLeading && !reachedNonWhite) || lastWasWhite) continue;
        accum.append(' ');
        lastWasWhite = true;
      } else if (!isInvisibleChar(c)) {
        accum.appendCodePoint(c);
        lastWasWhite = false;
        reachedNonWhite = true;
      }
    }
  }

  private static void appendNormalisedText(StringBuilder accum, TextNode textNode) {
    String text = textNode.getWholeText();
    if (preserveWhitespace(textNode.parentNode) || textNode instanceof CDataNode)
      accum.append(text);
    else StringUtil.appendNormalisedWhitespace(accum, text, TextNode.lastCharIsWhitespace(accum));
  }

  void resetInsertionMode() {
    boolean last = false;
    for (int pos = stack.size() - 1; pos >= 0; pos--) {
      Element node = stack.get(pos);
      if (pos == 0) {
        last = true;
        node = contextElement;
      }
      String name = node.normalName();
      if ("select".equals(name)) {
        transition(HtmlTreeBuilderState.InSelect);
        break;
      } else if (("td".equals(name) || "th".equals(name) && !last)) {
        transition(HtmlTreeBuilderState.InCell);
        break;
      } else if ("tr".equals(name)) {
        transition(HtmlTreeBuilderState.InRow);
        break;
      } else if ("tbody".equals(name) || "thead".equals(name) || "tfoot".equals(name)) {
        transition(HtmlTreeBuilderState.InTableBody);
        break;
      } else if ("caption".equals(name)) {
        transition(HtmlTreeBuilderState.InCaption);
        break;
      } else if ("colgroup".equals(name)) {
        transition(HtmlTreeBuilderState.InColumnGroup);
        break;
      } else if ("table".equals(name)) {
        transition(HtmlTreeBuilderState.InTable);
        break;
      } else if ("head".equals(name)) {
        transition(HtmlTreeBuilderState.InBody);
        break;
      } else if ("body".equals(name)) {
        transition(HtmlTreeBuilderState.InBody);
        break;
      } else if ("frameset".equals(name)) {
        transition(HtmlTreeBuilderState.InFrameset);
        break;
      } else if ("html".equals(name)) {
        transition(HtmlTreeBuilderState.BeforeHead);
        break;
      } else if (last) {
        transition(HtmlTreeBuilderState.InBody);
        break;
      }
    }
  }

  @Override
  Tag reset() {
    tagName = null;
    normalName = null;
    pendingAttributeName = null;
    reset(pendingAttributeValue);
    pendingAttributeValueS = null;
    hasEmptyAttributeValue = false;
    hasPendingAttributeValue = false;
    selfClosing = false;
    attributes = null;
    return this;
  }

  void putIgnoreCase(String key, String value) {
    int i = indexOfKeyIgnoreCase(key);
    if (i != NotFound) {
      vals[i] = value;
      if (!keys[i].equals(key)) keys[i] = key;
    } else add(key, value);
  }

  public void reset() throws IOException {
    super.reset();
    remaining = maxSize - markpos;
  }

  final void appendAttributeValue(int[] appendCodepoints) {
    ensureAttributeValue();
    for (int codepoint : appendCodepoints) {
      pendingAttributeValue.appendCodePoint(codepoint);
    }
  }

  @Override
  Token reset() {
    reset(name);
    pubSysKey = null;
    reset(publicIdentifier);
    reset(systemIdentifier);
    forceQuirks = false;
    return this;
  }

  Element aboveOnStack(Element el) {
    assert onStack(el);
    for (int pos = stack.size() - 1; pos >= 0; pos--) {
      Element next = stack.get(pos);
      if (next == el) {
        return stack.get(pos - 1);
      }
    }
    return null;
  }
}
