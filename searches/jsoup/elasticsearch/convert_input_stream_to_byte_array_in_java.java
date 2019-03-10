public class Results {

  public UnsafeByteArrayInputStream(byte buf[], int offset, int length) {
    mData = buf;
    mPosition = mMark = offset;
    mLimit = Math.min(offset + length, buf.length);
  }

  private static Object toArray(Class<?> c, Stack<Object> list, int len) throws ParseException {
    if (c == String.class) {
      if (len == 0) {
        return EMPTY_STRING_ARRAY;
      } else {
        Object o;
        String ss[] = new String[len];
        for (int i = len - 1; i >= 0; i--) {
          o = list.pop();
          ss[i] = (o == null ? null : o.toString());
        }
        return ss;
      }
    }
    if (c == boolean.class) {
      if (len == 0) {
        return EMPTY_BOOL_ARRAY;
      }
      Object o;
      boolean[] ret = new boolean[len];
      for (int i = len - 1; i >= 0; i--) {
        o = list.pop();
        if (o instanceof Boolean) {
          ret[i] = ((Boolean) o).booleanValue();
        }
      }
      return ret;
    }
    if (c == int.class) {
      if (len == 0) {
        return EMPTY_INT_ARRAY;
      }
      Object o;
      int[] ret = new int[len];
      for (int i = len - 1; i >= 0; i--) {
        o = list.pop();
        if (o instanceof Number) {
          ret[i] = ((Number) o).intValue();
        }
      }
      return ret;
    }
    if (c == long.class) {
      if (len == 0) {
        return EMPTY_LONG_ARRAY;
      }
      Object o;
      long[] ret = new long[len];
      for (int i = len - 1; i >= 0; i--) {
        o = list.pop();
        if (o instanceof Number) {
          ret[i] = ((Number) o).longValue();
        }
      }
      return ret;
    }
    if (c == float.class) {
      if (len == 0) {
        return EMPTY_FLOAT_ARRAY;
      }
      Object o;
      float[] ret = new float[len];
      for (int i = len - 1; i >= 0; i--) {
        o = list.pop();
        if (o instanceof Number) {
          ret[i] = ((Number) o).floatValue();
        }
      }
      return ret;
    }
    if (c == double.class) {
      if (len == 0) {
        return EMPTY_DOUBLE_ARRAY;
      }
      Object o;
      double[] ret = new double[len];
      for (int i = len - 1; i >= 0; i--) {
        o = list.pop();
        if (o instanceof Number) {
          ret[i] = ((Number) o).doubleValue();
        }
      }
      return ret;
    }
    if (c == byte.class) {
      if (len == 0) {
        return EMPTY_BYTE_ARRAY;
      }
      Object o;
      byte[] ret = new byte[len];
      for (int i = len - 1; i >= 0; i--) {
        o = list.pop();
        if (o instanceof Number) {
          ret[i] = ((Number) o).byteValue();
        }
      }
      return ret;
    }
    if (c == char.class) {
      if (len == 0) {
        return EMPTY_CHAR_ARRAY;
      }
      Object o;
      char[] ret = new char[len];
      for (int i = len - 1; i >= 0; i--) {
        o = list.pop();
        if (o instanceof Character) {
          ret[i] = ((Character) o).charValue();
        }
      }
      return ret;
    }
    if (c == short.class) {
      if (len == 0) {
        return EMPTY_SHORT_ARRAY;
      }
      Object o;
      short[] ret = new short[len];
      for (int i = len - 1; i >= 0; i--) {
        o = list.pop();
        if (o instanceof Number) {
          ret[i] = ((Number) o).shortValue();
        }
      }
      return ret;
    }
    Object ret = Array.newInstance(c, len);
    for (int i = len - 1; i >= 0; i--) {
      Array.set(ret, i, list.pop());
    }
    return ret;
  }

  public ByteBuffer toByteBuffer(int index, int length) {
    return ByteBuffer.wrap(array, index, length);
  }

  default List<URL> convert(URL subscribeUrl, Object source) {
    return this.convert(new com.alibaba.dubbo.common.URL(subscribeUrl), source).stream()
        .map(url -> url.getOriginalURL())
        .collect(Collectors.toList());
  }

  public byte[] array() {
    return buffer.array();
  }

  public byte[] array() {
    return buffer.array();
  }

  public byte[] array() {
    return buffer.array();
  }

  public byte[] array() {
    return array;
  }

  public byte[] array() {
    return buffer.array();
  }

  default <T> T convert(Class<T> cls, String key, T defaultValue) {
    String value = (String) getProperty(key);
    if (value == null) {
      return defaultValue;
    }
    Object obj = value;
    if (cls.isInstance(value)) {
      return cls.cast(value);
    }
    if (String.class.equals(cls)) {
      return cls.cast(value);
    }
    if (Boolean.class.equals(cls) || Boolean.TYPE.equals(cls)) {
      obj = Boolean.valueOf(value);
    } else if (Number.class.isAssignableFrom(cls) || cls.isPrimitive()) {
      if (Integer.class.equals(cls) || Integer.TYPE.equals(cls)) {
        obj = Integer.valueOf(value);
      } else if (Long.class.equals(cls) || Long.TYPE.equals(cls)) {
        obj = Long.valueOf(value);
      } else if (Byte.class.equals(cls) || Byte.TYPE.equals(cls)) {
        obj = Byte.valueOf(value);
      } else if (Short.class.equals(cls) || Short.TYPE.equals(cls)) {
        obj = Short.valueOf(value);
      } else if (Float.class.equals(cls) || Float.TYPE.equals(cls)) {
        obj = Float.valueOf(value);
      } else if (Double.class.equals(cls) || Double.TYPE.equals(cls)) {
        obj = Double.valueOf(value);
      }
    } else if (cls.isEnum()) {
      obj = Enum.valueOf(cls.asSubclass(Enum.class), value);
    }
    return cls.cast(obj);
  }
}
