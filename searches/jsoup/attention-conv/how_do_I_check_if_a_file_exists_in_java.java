public class Results {

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

  private void characterReferenceError(String message) {
    if (errors.canAddError())
      errors.add(new ParseError(reader.pos(), "invalid character reference: %s", message));
  }

  public Element val(String value) {
    if (tagName().equals("textarea")) text(value);
    else attr("value", value);
    return this;
  }

  int codepointForName(final String name) {
    int index = Arrays.binarySearch(nameKeys, name);
    return index >= 0 ? codeVals[index] : empty;
  }

  public void setPubSysKey(String value) {
    if (value != null) attr(PUB_SYS_KEY, value);
  }

  public boolean matches(Element root, Element element) {
    return element.hasAttr(key);
  }

  public String val() {
    if (tagName().equals("textarea")) return text();
    else return attr("value");
  }

  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((value == null) ? 0 : value.hashCode());
    return result;
  }
}
