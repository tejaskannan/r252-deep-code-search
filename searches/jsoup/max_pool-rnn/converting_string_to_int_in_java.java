public class Results {

  public static String unescape(String in) {
    StringBuilder out = StringUtil.borrowBuilder();
    char last = 0;
    for (char c : in.toCharArray()) {
      if (c == ESC) {
        if (last != 0 && last == ESC) out.append(c);
      } else out.append(c);
      last = c;
    }
    return StringUtil.releaseBuilder(out);
  }

  void putIgnoreCase(String key, String value) {
    int i = indexOfKeyIgnoreCase(key);
    if (i != NotFound) {
      vals[i] = value;
      if (!keys[i].equals(key)) keys[i] = key;
    } else add(key, value);
  }

  public Element appendElement(String tagName) {
    Element child = new Element(Tag.valueOf(tagName, NodeUtils.parser(this).settings()), baseUri());
    appendChild(child);
    return child;
  }

  public String normalizeAttribute(String name) {
    name = name.trim();
    if (!preserveAttributeCase) name = lowerCase(name);
    return name;
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

  private static void appendWhitespaceIfBr(Element element, StringBuilder accum) {
    if (element.tag.getName().equals("br") && !TextNode.lastCharIsWhitespace(accum))
      accum.append(" ");
  }

  public static String escape(String string, Document.OutputSettings out) {
    if (string == null) return "";
    StringBuilder accum = StringUtil.borrowBuilder();
    try {
      escape(accum, string, out, false, false, false);
    } catch (IOException e) {
      throw new SerializationException(e);
    }
    return StringUtil.releaseBuilder(accum);
  }

  boolean matchesIgnoreCase(String seq) {
    bufferUp();
    int scanLength = seq.length();
    if (scanLength > bufLength - bufPos) return false;
    for (int offset = 0; offset < scanLength; offset++) {
      char upScan = Character.toUpperCase(seq.charAt(offset));
      char upTarget = Character.toUpperCase(charBuf[bufPos + offset]);
      if (upScan != upTarget) return false;
    }
    return true;
  }

  public Attribute(String key, String val, Attributes parent) {
    Validate.notNull(key);
    key = key.trim();
    Validate.notEmpty(key);
    this.key = key;
    this.val = val;
    this.parent = parent;
  }

  private static void appendNormalisedText(StringBuilder accum, TextNode textNode) {
    String text = textNode.getWholeText();
    if (preserveWhitespace(textNode.parentNode) || textNode instanceof CDataNode)
      accum.append(text);
    else StringUtil.appendNormalisedWhitespace(accum, text, TextNode.lastCharIsWhitespace(accum));
  }
}
