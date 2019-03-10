public class Results {

  public Element attr(String attributeKey, String attributeValue) {
    super.attr(attributeKey, attributeValue);
    return this;
  }

  public Elements attr(String attributeKey, String attributeValue) {
    for (Element element : this) {
      element.attr(attributeKey, attributeValue);
    }
    return this;
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

  CharsetEncoder prepareEncoder() {
    CharsetEncoder encoder = charset.newEncoder();
    encoderThreadLocal.set(encoder);
    coreCharset = Entities.CoreCharset.byName(encoder.charset().name());
    return encoder;
  }

  public int hashCode() {
    int result = key != null ? key.hashCode() : 0;
    result = 31 * result + (val != null ? val.hashCode() : 0);
    return result;
  }

  public Iterator<Attribute> iterator() {
    return new Iterator<Attribute>() {
      int i = 0;

      @Override
      public boolean hasNext() {
        return i < size;
      }

      @Override
      public Attribute next() {
        final Attribute attr = new Attribute(keys[i], vals[i], Attributes.this);
        i++;
        return attr;
      }

      @Override
      public void remove() {
        Attributes.this.remove(--i);
      }
    };
  }

  public String attr(String attributeKey) {
    for (Element element : this) {
      if (element.hasAttr(attributeKey)) return element.attr(attributeKey);
    }
    return "";
  }

  public Element val(String value) {
    if (tagName().equals("textarea")) text(value);
    else attr("value", value);
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
}
