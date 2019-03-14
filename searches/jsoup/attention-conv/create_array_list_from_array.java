public class Results {

  public int hashCode() {
    int result = size;
    result = 31 * result + Arrays.hashCode(keys);
    result = 31 * result + Arrays.hashCode(vals);
    return result;
  }

  public Elements attr(String attributeKey, String attributeValue) {
    for (Element element : this) {
      element.attr(attributeKey, attributeValue);
    }
    return this;
  }

  public DocumentType(String name, String publicId, String systemId) {
    Validate.notNull(name);
    Validate.notNull(publicId);
    Validate.notNull(systemId);
    attr(NAME, name);
    attr(PUBLIC_ID, publicId);
    if (has(PUBLIC_ID)) {
      attr(PUB_SYS_KEY, PUBLIC_KEY);
    }
    attr(SYSTEM_ID, systemId);
  }

  public DocumentType(String name, String publicId, String systemId, String baseUri) {
    attr(NAME, name);
    attr(PUBLIC_ID, publicId);
    if (has(PUBLIC_ID)) {
      attr(PUB_SYS_KEY, PUBLIC_KEY);
    }
    attr(SYSTEM_ID, systemId);
  }

  public DocumentType(
      String name, String pubSysKey, String publicId, String systemId, String baseUri) {
    attr(NAME, name);
    if (pubSysKey != null) {
      attr(PUB_SYS_KEY, pubSysKey);
    }
    attr(PUBLIC_ID, publicId);
    attr(SYSTEM_ID, systemId);
  }

  public int hashCode() {
    int result = tagName.hashCode();
    result = 31 * result + (isBlock ? 1 : 0);
    result = 31 * result + (formatAsBlock ? 1 : 0);
    result = 31 * result + (canContainInline ? 1 : 0);
    result = 31 * result + (empty ? 1 : 0);
    result = 31 * result + (selfClosing ? 1 : 0);
    result = 31 * result + (preserveWhitespace ? 1 : 0);
    result = 31 * result + (formList ? 1 : 0);
    result = 31 * result + (formSubmit ? 1 : 0);
    return result;
  }

  public Element attr(String attributeKey, String attributeValue) {
    super.attr(attributeKey, attributeValue);
    return this;
  }

  public String attr(String attributeKey) {
    for (Element element : this) {
      if (element.hasAttr(attributeKey)) return element.attr(attributeKey);
    }
    return "";
  }

  CharsetEncoder prepareEncoder() {
    CharsetEncoder encoder = charset.newEncoder();
    encoderThreadLocal.set(encoder);
    coreCharset = Entities.CoreCharset.byName(encoder.charset().name());
    return encoder;
  }

  public XmlDeclaration(String name, boolean isProcessingInstruction) {
    Validate.notNull(name);
    value = name;
    this.isProcessingInstruction = isProcessingInstruction;
  }
}
