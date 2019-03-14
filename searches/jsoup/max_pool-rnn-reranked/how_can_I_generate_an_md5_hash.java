public class Results {

  private Base() {
    headers = new LinkedHashMap<>();
    cookies = new LinkedHashMap<>();
  }

  public boolean matches(Element root, Element element) {
    return !value.equalsIgnoreCase(element.attr(key));
  }

  private static String validateCharset(String cs) {
    if (cs == null || cs.length() == 0) return null;
    cs = cs.trim().replaceAll("[\"\']", "");
    try {
      if (Charset.isSupported(cs)) return cs;
      cs = cs.toUpperCase(Locale.ENGLISH);
      if (Charset.isSupported(cs)) return cs;
    } catch (IllegalCharsetNameException e) {
    }
    return null;
  }

  protected Element doClone(Node parent) {
    Element clone = (Element) super.doClone(parent);
    clone.attributes = attributes != null ? attributes.clone() : null;
    clone.baseUri = baseUri;
    clone.childNodes = new NodeList(clone, childNodes.size());
    clone.childNodes.addAll(childNodes);
    return clone;
  }

  protected Node doClone(Node parent) {
    Node clone;
    try {
      clone = (Node) super.clone();
    } catch (CloneNotSupportedException e) {
      throw new RuntimeException(e);
    }
    clone.parentNode = parent;
    clone.siblingIndex = parent == null ? 0 : siblingIndex;
    return clone;
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

  protected void runParser() {
    while (true) {
      Token token = tokeniser.read();
      process(token);
      token.reset();
      if (token.type == Token.TokenType.EOF) break;
    }
  }

  private static String encodeUrl(String url) {
    try {
      URL u = new URL(url);
      return encodeUrl(u).toExternalForm();
    } catch (Exception e) {
      return url;
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

  public Attribute(String key, String val, Attributes parent) {
    Validate.notNull(key);
    key = key.trim();
    Validate.notEmpty(key);
    this.key = key;
    this.val = val;
    this.parent = parent;
  }
}
