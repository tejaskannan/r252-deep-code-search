public class Results {

  public Stream<String> visitArray(List<? extends AnnotationValue> list, Void unused) {
    return list.stream().flatMap(a -> a.accept(this, null)).filter(x -> x != null);
  }

  public byte[] array() {
    return array;
  }

  public byte[] array() {
    return buffer.array();
  }

  public Void visitArray(List<? extends AnnotationValue> list, Void unused) {
    for (AnnotationValue value : list) {
      value.accept(this, null);
    }
    return null;
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

  public Stream<String> visitArray(List<? extends AnnotationValue> list, Void aVoid) {
    return list.stream().flatMap(a -> a.accept(this, null)).filter(x -> x != null);
  }

  public Void visitArray(List<? extends AnnotationValue> vals, Void p) {
    for (AnnotationValue val : vals) {
      visit(val);
    }
    return null;
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
}
