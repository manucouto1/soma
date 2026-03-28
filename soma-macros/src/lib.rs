use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields, Lit, Expr};

/// Derive macro for generating `config_hash()` and `Searchable` implementations.
///
/// # Attributes
///
/// ## Struct-level
/// - `#[soma(kind = "Trainable")]` - FilterKind (Stateless, Trainable, Opaque)
/// - `#[soma(cacheable)]` or `#[soma(cacheable = false)]`
/// - `#[soma(differentiable)]` or `#[soma(differentiable = false)]`
/// - `#[soma(stream = "FixedState")]` - StreamMode
///
/// ## Field-level
/// - `#[soma(search(low = 0.1, high = 10.0))]` - Float search range
/// - `#[soma(search(low = 0.1, high = 10.0, scale = "log"))]` - With scale
/// - `#[soma(search(choices = ["a", "b", "c"]))]` - Categorical
/// - `#[soma(search)]` - Auto-detect from type (bool → Categorical)
/// - `#[soma(skip_hash)]` - Exclude from config_hash
#[proc_macro_derive(SomaFilter, attributes(soma))]
pub fn derive_soma_filter(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let name_str = name.to_string();

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("SomaFilter only supports structs with named fields"),
        },
        _ => panic!("SomaFilter only supports structs"),
    };

    // Parse struct-level attributes
    let struct_attrs = parse_struct_attrs(&input.attrs);

    // Separate fields into hash fields and search fields
    let mut hash_parts = Vec::new();
    let mut search_dims = Vec::new();
    let mut from_sample_fields = Vec::new();
    let mut current_params_fields = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();
        let field_ty = &field.ty;
        let field_attrs = parse_field_attrs(&field.attrs);

        // config_hash: include unless skip_hash
        if !field_attrs.skip_hash {
            hash_parts.push(quote! {
                {
                    let serialized = serde_json::to_vec(&self.#field_name)
                        .unwrap_or_default();
                    parts.push(serialized);
                }
            });
        }

        // search: generate SearchDimension if annotated
        if let Some(search) = &field_attrs.search {
            let dim = generate_search_dimension(&field_name_str, field_ty, search);
            search_dims.push(dim);

            // from_sample: extract from params map
            from_sample_fields.push(generate_from_sample(field_name, &field_name_str, field_ty));

            // current_params: insert into map
            current_params_fields.push(quote! {
                params.insert(
                    #field_name_str.to_string(),
                    serde_json::to_value(&self.#field_name).unwrap_or_default(),
                );
            });
        } else {
            // Non-searchable: use Default
            from_sample_fields.push(quote! {
                #field_name: Default::default(),
            });
        }
    }

    // FilterKind
    let kind = match struct_attrs.kind.as_deref() {
        Some("Stateless") => quote! { soma_core::filter::FilterKind::Stateless },
        Some("Opaque") => quote! { soma_core::filter::FilterKind::Opaque },
        _ => quote! { soma_core::filter::FilterKind::Trainable },
    };

    let cacheable = struct_attrs.cacheable;
    let differentiable = struct_attrs.differentiable;

    let stream_mode = match struct_attrs.stream.as_deref() {
        Some("Barrier") => quote! { soma_core::filter::StreamMode::Barrier },
        Some("Evolving") => quote! { soma_core::filter::StreamMode::Evolving { checkpoint_every: 100 } },
        _ => quote! { soma_core::filter::StreamMode::FixedState },
    };

    let expanded = quote! {
        impl #name {
            /// Compute a content-addressable hash of this filter's configuration.
            pub fn config_hash(&self) -> soma_core::cache::CacheKey {
                let mut parts: Vec<Vec<u8>> = Vec::new();
                // Include type name
                parts.push(#name_str.as_bytes().to_vec());
                // Include each non-skipped field
                #(#hash_parts)*
                let refs: Vec<&[u8]> = parts.iter().map(|p| p.as_slice()).collect();
                soma_core::cache::CacheKey::from_parts(&refs)
            }

            /// Filter metadata for the compiler.
            pub fn soma_meta(&self) -> soma_core::filter::FilterMeta {
                soma_core::filter::FilterMeta {
                    name: #name_str.to_string(),
                    kind: #kind,
                    cacheable: #cacheable,
                    differentiable: #differentiable,
                    stream_mode: #stream_mode,
                }
            }
        }

        impl soma_core::search::Searchable for #name {
            fn search_space() -> soma_core::search::SearchSpace {
                let mut space = soma_core::search::SearchSpace::new();
                #(#search_dims)*
                space
            }

            fn from_sample(
                params: &std::collections::HashMap<String, serde_json::Value>,
            ) -> soma_core::error::Result<Self> {
                Ok(Self {
                    #(#from_sample_fields)*
                })
            }

            fn current_params(&self) -> std::collections::HashMap<String, serde_json::Value> {
                let mut params = std::collections::HashMap::new();
                #(#current_params_fields)*
                params
            }
        }
    };

    TokenStream::from(expanded)
}

// ── Attribute parsing ──

struct StructAttrs {
    kind: Option<String>,
    cacheable: bool,
    differentiable: bool,
    stream: Option<String>,
}

struct FieldAttrs {
    skip_hash: bool,
    search: Option<SearchAttrs>,
}

struct SearchAttrs {
    low: Option<f64>,
    high: Option<f64>,
    scale: Option<String>,
    choices: Vec<String>,
    auto: bool, // #[soma(search)] with no args
}

fn parse_struct_attrs(attrs: &[syn::Attribute]) -> StructAttrs {
    let mut result = StructAttrs {
        kind: None,
        cacheable: true,
        differentiable: true,
        stream: None,
    };

    for attr in attrs {
        if !attr.path().is_ident("soma") {
            continue;
        }
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("kind") {
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                if let Lit::Str(s) = lit {
                    result.kind = Some(s.value());
                }
            } else if meta.path.is_ident("cacheable") {
                if let Ok(value) = meta.value() {
                    let lit: Lit = value.parse()?;
                    if let Lit::Bool(b) = lit {
                        result.cacheable = b.value;
                    }
                }
                // else: just #[soma(cacheable)] means true
            } else if meta.path.is_ident("differentiable") {
                if let Ok(value) = meta.value() {
                    let lit: Lit = value.parse()?;
                    if let Lit::Bool(b) = lit {
                        result.differentiable = b.value;
                    }
                }
            } else if meta.path.is_ident("stream") {
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                if let Lit::Str(s) = lit {
                    result.stream = Some(s.value());
                }
            }
            Ok(())
        });
    }

    result
}

fn parse_field_attrs(attrs: &[syn::Attribute]) -> FieldAttrs {
    let mut result = FieldAttrs {
        skip_hash: false,
        search: None,
    };

    for attr in attrs {
        if !attr.path().is_ident("soma") {
            continue;
        }
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip_hash") {
                result.skip_hash = true;
            } else if meta.path.is_ident("search") {
                let mut search = SearchAttrs {
                    low: None,
                    high: None,
                    scale: None,
                    choices: Vec::new(),
                    auto: false,
                };

                if meta.input.peek(syn::token::Paren) {
                    let _ = meta.parse_nested_meta(|inner| {
                        if inner.path.is_ident("low") {
                            let value = inner.value()?;
                            let expr: Expr = value.parse()?;
                            search.low = Some(expr_to_f64(&expr));
                        } else if inner.path.is_ident("high") {
                            let value = inner.value()?;
                            let expr: Expr = value.parse()?;
                            search.high = Some(expr_to_f64(&expr));
                        } else if inner.path.is_ident("scale") {
                            let value = inner.value()?;
                            let lit: Lit = value.parse()?;
                            if let Lit::Str(s) = lit {
                                search.scale = Some(s.value());
                            }
                        } else if inner.path.is_ident("choices") {
                            let value = inner.value()?;
                            let array: syn::ExprArray = value.parse()?;
                            for elem in &array.elems {
                                if let Expr::Lit(lit) = elem {
                                    match &lit.lit {
                                        Lit::Str(s) => search.choices.push(s.value()),
                                        Lit::Int(i) => search.choices.push(i.to_string()),
                                        Lit::Float(f) => search.choices.push(f.to_string()),
                                        Lit::Bool(b) => search.choices.push(b.value.to_string()),
                                        _ => {}
                                    }
                                }
                            }
                        }
                        Ok(())
                    });
                } else {
                    search.auto = true;
                }

                result.search = Some(search);
            }
            Ok(())
        });
    }

    result
}

fn expr_to_f64(expr: &Expr) -> f64 {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            Lit::Float(f) => f.base10_parse().unwrap(),
            Lit::Int(i) => i.base10_parse::<i64>().unwrap() as f64,
            _ => panic!("expected numeric literal"),
        },
        Expr::Unary(unary) => {
            if matches!(unary.op, syn::UnOp::Neg(_)) {
                -expr_to_f64(&unary.expr)
            } else {
                panic!("unsupported unary op")
            }
        }
        _ => panic!("expected numeric literal, got {:?}", expr),
    }
}

// ── Code generation helpers ──

fn generate_search_dimension(
    name: &str,
    ty: &syn::Type,
    search: &SearchAttrs,
) -> proc_macro2::TokenStream {
    let type_str = quote!(#ty).to_string();

    // Categorical from choices
    if !search.choices.is_empty() {
        let choices = &search.choices;
        return quote! {
            space.add(soma_core::search::SearchDimension::Categorical {
                name: #name.to_string(),
                choices: vec![#(serde_json::json!(#choices)),*],
            });
        };
    }

    // Auto-detect for bool
    if search.auto && type_str == "bool" {
        return quote! {
            space.add(soma_core::search::SearchDimension::Categorical {
                name: #name.to_string(),
                choices: vec![serde_json::json!(true), serde_json::json!(false)],
            });
        };
    }

    // Float range
    if let (Some(low), Some(high)) = (search.low, search.high) {
        let scale = match search.scale.as_deref() {
            Some("log") => quote! { soma_core::search::Scale::Log },
            Some("reverse_log") => quote! { soma_core::search::Scale::ReverseLog },
            _ => quote! { soma_core::search::Scale::Linear },
        };

        let is_int = type_str.contains("usize")
            || type_str.contains("i32")
            || type_str.contains("i64")
            || type_str.contains("u32")
            || type_str.contains("u64");

        if is_int {
            let low_i = low as i64;
            let high_i = high as i64;
            return quote! {
                space.add(soma_core::search::SearchDimension::Int {
                    name: #name.to_string(),
                    low: #low_i,
                    high: #high_i,
                    scale: #scale,
                });
            };
        } else {
            return quote! {
                space.add(soma_core::search::SearchDimension::Float {
                    name: #name.to_string(),
                    low: #low,
                    high: #high,
                    scale: #scale,
                    default: None,
                });
            };
        }
    }

    // Fallback: auto bool already handled, no search for unknown types
    quote! {}
}

fn generate_from_sample(
    field_name: &syn::Ident,
    field_name_str: &str,
    field_ty: &syn::Type,
) -> proc_macro2::TokenStream {
    let type_str = quote!(#field_ty).to_string();

    if type_str.contains("String") {
        quote! {
            #field_name: params.get(#field_name_str)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
        }
    } else if type_str == "bool" {
        quote! {
            #field_name: params.get(#field_name_str)
                .and_then(|v| v.as_bool())
                .unwrap_or_default(),
        }
    } else if type_str.contains("usize") {
        quote! {
            #field_name: params.get(#field_name_str)
                .and_then(|v| v.as_u64())
                .unwrap_or_default() as usize,
        }
    } else if type_str.contains("i64") || type_str.contains("i32") {
        quote! {
            #field_name: params.get(#field_name_str)
                .and_then(|v| v.as_i64())
                .unwrap_or_default() as #field_ty,
        }
    } else {
        // f64, f32
        quote! {
            #field_name: params.get(#field_name_str)
                .and_then(|v| v.as_f64())
                .unwrap_or_default() as #field_ty,
        }
    }
}
