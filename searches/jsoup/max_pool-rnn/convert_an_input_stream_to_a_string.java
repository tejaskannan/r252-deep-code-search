public class Results {

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

  private static void appendWhitespaceIfBr(Element element, StringBuilder accum) {
    if (element.tag.getName().equals("br") && !TextNode.lastCharIsWhitespace(accum))
      accum.append(" ");
  }

  public Elements parents() {
    HashSet<Element> combo = new LinkedHashSet<>();
    for (Element e : this) {
      combo.addAll(e.parents());
    }
    return new Elements(combo);
  }

  public Element appendElement(String tagName) {
    Element child = new Element(Tag.valueOf(tagName, NodeUtils.parser(this).settings()), baseUri());
    appendChild(child);
    return child;
  }

  public String cssSelector() {
    if (id().length() > 0) return "#" + id();
    String tagName = tagName().replace(':', '|');
    StringBuilder selector = new StringBuilder(tagName);
    String classes = StringUtil.join(classNames(), ".");
    if (classes.length() > 0) selector.append('.').append(classes);
    if (parent() == null || parent() instanceof Document) return selector.toString();
    selector.insert(0, " > ");
    if (parent().select(selector.toString()).size() > 1)
      selector.append(String.format(":nth-child(%d)", elementSiblingIndex() + 1));
    return parent().cssSelector() + selector.toString();
  }

  public byte[] bodyAsBytes() {
    prepareByteData();
    return byteData.array();
  }

  private static void appendNormalisedText(StringBuilder accum, TextNode textNode) {
    String text = textNode.getWholeText();
    if (preserveWhitespace(textNode.parentNode) || textNode instanceof CDataNode)
      accum.append(text);
    else StringUtil.appendNormalisedWhitespace(accum, text, TextNode.lastCharIsWhitespace(accum));
  }

  private static String fixHeaderEncoding(String val) {
    try {
      byte[] bytes = val.getBytes("iso-8859-1");
      if (!looksLikeUtf8(bytes)) return val;
      return new String(bytes, "utf-8");
    } catch (UnsupportedEncodingException e) {
      return val;
    }
  }

  public Node clearAttributes() {
    Iterator<Attribute> it = attributes().iterator();
    while (it.hasNext()) {
      it.next();
      it.remove();
    }
    return this;
  }
}
