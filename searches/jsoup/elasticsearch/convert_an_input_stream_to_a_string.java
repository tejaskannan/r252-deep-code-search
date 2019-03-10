public class Results {

  default List<URL> convert(URL subscribeUrl, Object source) {
    return this.convert(new com.alibaba.dubbo.common.URL(subscribeUrl), source).stream()
        .map(url -> url.getOriginalURL())
        .collect(Collectors.toList());
  }

  private static String convertToString(ExpressionTree detail, VisitorState state) {
    return state.getSourceForNode(detail)
        + (ASTHelpers.isSameType(ASTHelpers.getType(detail), state.getSymtab().stringType, state)
            ? ""
            : ".tostring()");
  }

  public Target convert(String arg) {
    return Target.valueOf(Ascii.toUpperCase(arg));
  }

  private void convertProtocolIdsToProtocols() {
    if (StringUtils.isEmpty(protocolIds) && CollectionUtils.isEmpty(protocols)) {
      List<String> configedProtocols = new ArrayList<>();
      configedProtocols.addAll(
          getSubProperties(
              Environment.getInstance().getExternalConfigurationMap(), Constants.PROTOCOLS_SUFFIX));
      configedProtocols.addAll(
          getSubProperties(
              Environment.getInstance().getAppExternalConfigurationMap(),
              Constants.PROTOCOLS_SUFFIX));
      protocolIds = String.join(",", configedProtocols);
    }
    if (StringUtils.isEmpty(protocolIds)) {
      if (CollectionUtils.isEmpty(protocols)) {
        setProtocols(
            ConfigManager.getInstance()
                .getDefaultProtocols()
                .filter(CollectionUtils::isNotEmpty)
                .orElseGet(
                    () -> {
                      ProtocolConfig protocolConfig = new ProtocolConfig();
                      protocolConfig.refresh();
                      return Arrays.asList(protocolConfig);
                    }));
      }
    } else {
      String[] arr = Constants.COMMA_SPLIT_PATTERN.split(protocolIds);
      List<ProtocolConfig> tmpProtocols =
          CollectionUtils.isNotEmpty(protocols) ? protocols : new ArrayList<>();
      Arrays.stream(arr)
          .forEach(
              id -> {
                if (tmpProtocols.stream().noneMatch(prot -> prot.getId().equals(id))) {
                  tmpProtocols.add(
                      ConfigManager.getInstance()
                          .getProtocol(id)
                          .orElseGet(
                              () -> {
                                ProtocolConfig protocolConfig = new ProtocolConfig();
                                protocolConfig.setId(id);
                                protocolConfig.refresh();
                                return protocolConfig;
                              }));
                }
              });
      if (tmpProtocols.size() > arr.length) {
        throw new IllegalStateException(
            "too much protocols found, the protocols comply to this service are :"
                + protocolIds
                + " but got "
                + protocols.size()
                + " registries!");
      }
      setProtocols(tmpProtocols);
    }
  }

  private void convertRegistryIdsToRegistries() {
    if (StringUtils.isEmpty(registryIds) && CollectionUtils.isEmpty(registries)) {
      Set<String> configedRegistries = new HashSet<>();
      configedRegistries.addAll(
          getSubProperties(
              Environment.getInstance().getExternalConfigurationMap(),
              Constants.REGISTRIES_SUFFIX));
      configedRegistries.addAll(
          getSubProperties(
              Environment.getInstance().getAppExternalConfigurationMap(),
              Constants.REGISTRIES_SUFFIX));
      registryIds = String.join(",", configedRegistries);
    }
    if (StringUtils.isEmpty(registryIds)) {
      if (CollectionUtils.isEmpty(registries)) {
        setRegistries(
            ConfigManager.getInstance()
                .getDefaultRegistries()
                .filter(CollectionUtils::isNotEmpty)
                .orElseGet(
                    () -> {
                      RegistryConfig registryConfig = new RegistryConfig();
                      registryConfig.refresh();
                      return Arrays.asList(registryConfig);
                    }));
      }
    } else {
      String[] ids = Constants.COMMA_SPLIT_PATTERN.split(registryIds);
      List<RegistryConfig> tmpRegistries =
          CollectionUtils.isNotEmpty(registries) ? registries : new ArrayList<>();
      Arrays.stream(ids)
          .forEach(
              id -> {
                if (tmpRegistries.stream().noneMatch(reg -> reg.getId().equals(id))) {
                  tmpRegistries.add(
                      ConfigManager.getInstance()
                          .getRegistry(id)
                          .orElseGet(
                              () -> {
                                RegistryConfig registryConfig = new RegistryConfig();
                                registryConfig.setId(id);
                                registryConfig.refresh();
                                return registryConfig;
                              }));
                }
              });
      if (tmpRegistries.size() > ids.length) {
        throw new IllegalStateException(
            "too much registries found, the registries assigned to this service "
                + "are :"
                + registryIds
                + ", but got "
                + tmpRegistries.size()
                + " registries!");
      }
      setRegistries(tmpRegistries);
    }
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

  private Description makeDescription(String rejectedJUnit4, Tree tree) {
    Description.Builder builder =
        buildDescription(tree)
            .setMessage(
                String.format(
                    "%s cannot be used inside a junit3 class. convert your class to junit4 style.",
                    rejectedJUnit4));
    return builder.build();
  }

  private static String messageForAnnos(List<AnnotationTree> annotationTrees) {
    String annoNames =
        annotationTrees.stream()
            .map(a -> Signatures.prettyType(ASTHelpers.getType(a)))
            .collect(Collectors.joining(" and "));
    return annoNames + " can only be applied to static methods.";
  }

  private static ProviderConfig convertProtocolToProvider(ProtocolConfig protocol) {
    ProviderConfig provider = new ProviderConfig();
    provider.setProtocol(protocol);
    provider.setServer(protocol.getServer());
    provider.setClient(protocol.getClient());
    provider.setCodec(protocol.getCodec());
    provider.setHost(protocol.getHost());
    provider.setPort(protocol.getPort());
    provider.setPath(protocol.getPath());
    provider.setPayload(protocol.getPayload());
    provider.setThreads(protocol.getThreads());
    provider.setParameters(protocol.getParameters());
    return provider;
  }

  private static List<ProviderConfig> convertProtocolToProvider(List<ProtocolConfig> protocols) {
    if (CollectionUtils.isEmpty(protocols)) {
      return null;
    }
    List<ProviderConfig> providers = new ArrayList<ProviderConfig>(protocols.size());
    for (ProtocolConfig provider : protocols) {
      providers.add(convertProtocolToProvider(provider));
    }
    return providers;
  }
}
