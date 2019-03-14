public class Results {

  public List<String> eachAttr(String attributeKey) {
    List<String> attrs = new ArrayList<>(size());
    for (Element element : this) {
      if (element.hasAttr(attributeKey)) attrs.add(element.attr(attributeKey));
    }
    return attrs;
  }

  public Node attr(String attributeKey, String attributeValue) {
    attributeKey = NodeUtils.parser(this).settings().normalizeAttribute(attributeKey);
    attributes().putIgnoreCase(attributeKey, attributeValue);
    return this;
  }

  public <T extends Appendable> T html(T appendable) {
    final int size = childNodes.size();
    for (int i = 0; i < size; i++) childNodes.get(i).outerHtml(appendable);
    return appendable;
  }

  public Elements clone() {
    Elements clone = new Elements(size());
    for (Element e : this) clone.add(e.clone());
    return clone;
  }

  public Element attr(String attributeKey, boolean attributeValue) {
    attributes().put(attributeKey, attributeValue);
    return this;
  }

  String consumeToEnd() {
    bufferUp();
    String data = cacheString(charBuf, stringCache, bufPos, bufLength - bufPos);
    bufPos = bufLength;
    return data;
  }

  public Elements attr(String attributeKey, String attributeValue) {
    for (Element element : this) {
      element.attr(attributeKey, attributeValue);
    }
    return this;
  }

  public String consumeToAny(final char... chars) {
    bufferUp();
    int pos = bufPos;
    final int start = pos;
    final int remaining = bufLength;
    final char[] val = charBuf;
    final int charLen = chars.length;
    int i;
    OUTER:
    while (pos < remaining) {
      for (i = 0; i < charLen; i++) {
        if (val[pos] == chars[i]) break OUTER;
      }
      pos++;
    }
    bufPos = pos;
    return pos > start ? cacheString(charBuf, stringCache, start, pos - start) : "";
  }

  public String attr(String attributeKey) {
    for (Element element : this) {
      if (element.hasAttr(attributeKey)) return element.attr(attributeKey);
    }
    return "";
  }

  public Element prependChild(Node child) {
    Validate.notNull(child);
    addChildren(0, child);
    return this;
  }
}
