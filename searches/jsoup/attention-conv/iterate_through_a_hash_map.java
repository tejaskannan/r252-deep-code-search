public class Results {

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

  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((value == null) ? 0 : value.hashCode());
    return result;
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

  public int hashCode() {
    int result = size;
    result = 31 * result + Arrays.hashCode(keys);
    result = 31 * result + Arrays.hashCode(vals);
    return result;
  }

  public int hashCode() {
    int result = key != null ? key.hashCode() : 0;
    result = 31 * result + (val != null ? val.hashCode() : 0);
    return result;
  }

  public Request proxy(String host, int port) {
    this.proxy = new Proxy(Proxy.Type.HTTP, InetSocketAddress.createUnresolved(host, port));
    return this;
  }
}
