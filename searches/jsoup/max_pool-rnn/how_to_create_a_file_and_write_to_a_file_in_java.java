public class Results {

  public Element classNames(Set<String> classNames) {
    Validate.notNull(classNames);
    if (classNames.isEmpty()) {
      attributes().remove("class");
    } else {
      attributes().put("class", StringUtil.join(classNames, " "));
    }
    return this;
  }

  private ConstrainableInputStream(InputStream in, int bufferSize, int maxSize) {
    super(in, bufferSize);
    Validate.isTrue(maxSize >= 0);
    this.maxSize = maxSize;
    remaining = maxSize;
    capped = maxSize != 0;
    startTime = System.nanoTime();
  }

  void reconstructFormattingElements() {
    Element last = lastFormattingElement();
    if (last == null || onStack(last)) return;
    Element entry = last;
    int size = formattingElements.size();
    int pos = size - 1;
    boolean skip = false;
    while (true) {
      if (pos == 0) {
        skip = true;
        break;
      }
      entry = formattingElements.get(--pos);
      if (entry == null || onStack(entry)) break;
    }
    while (true) {
      if (!skip) entry = formattingElements.get(++pos);
      Validate.notNull(entry);
      skip = false;
      Element newEl = insertStartTag(entry.normalName());
      newEl.attributes().addAll(entry.attributes());
      formattingElements.set(pos, newEl);
      if (pos == size - 1) break;
    }
  }

  private void clearStackToContext(String... nodeNames) {
    for (int pos = stack.size() - 1; pos >= 0; pos--) {
      Element next = stack.get(pos);
      if (StringUtil.in(next.normalName(), nodeNames) || next.normalName().equals("html")) break;
      else stack.remove(pos);
    }
  }

  public void charset(Charset charset) {
    updateMetaCharsetElement(true);
    outputSettings.charset(charset);
    ensureMetaCharsetElement();
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

  public Connection.Request postDataCharset(String charset) {
    Validate.notNull(charset, "charset must not be null");
    if (!Charset.isSupported(charset)) throw new IllegalCharsetNameException(charset);
    this.postDataCharset = charset;
    return this;
  }

  public byte[] bodyAsBytes() {
    prepareByteData();
    return byteData.array();
  }

  private ElementMeta createSafeElement(Element sourceEl) {
    String sourceTag = sourceEl.tagName();
    Attributes destAttrs = new Attributes();
    Element dest = new Element(Tag.valueOf(sourceTag), sourceEl.baseUri(), destAttrs);
    int numDiscarded = 0;
    Attributes sourceAttrs = sourceEl.attributes();
    for (Attribute sourceAttr : sourceAttrs) {
      if (whitelist.isSafeAttribute(sourceTag, sourceEl, sourceAttr)) destAttrs.put(sourceAttr);
      else numDiscarded++;
    }
    Attributes enforcedAttrs = whitelist.getEnforcedAttributes(sourceTag);
    destAttrs.addAll(enforcedAttrs);
    return new ElementMeta(dest, numDiscarded);
  }
}
