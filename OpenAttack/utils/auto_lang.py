def get_language(obj_list):
    lang_tag_cnt = {}
    for it in obj_list:
        for tag in it.TAGS:
            if tag.type == "lang":
                if tag not in lang_tag_cnt:
                    lang_tag_cnt[tag] = 0
                lang_tag_cnt[tag] += 1
    
    # argmax
    lang_tag = None
    for tag, cnt in lang_tag_cnt.items():
        if lang_tag is None or cnt > lang_tag_cnt[lang_tag]:
            lang_tag = tag

    # no language
    if lang_tag is None:
        raise RuntimeError("No language support")
    else:
        if lang_tag_cnt[lang_tag] < len(obj_list):
            unsupported_names = []
            for it in obj_list:
                if lang_tag not in it.TAGS:
                    unsupported_names.append( it.__class__.__name__ )
            raise RuntimeError("Try to use language `%s`, but %s not support " % (lang_tag, unsupported_names))
        else:
            return lang_tag

def check_language(obj_list, lang_tag):
    unsupported_names = []
    for it in obj_list:
        if lang_tag not in it.TAGS:
            unsupported_names.append( it.__class__.__name__ )
    if len(unsupported_names) > 0:
        raise RuntimeError("using language `%s`, but %s not support " % (lang_tag, unsupported_names))

def language_by_name(name):
    from ..tags import TAG_ALL_LANGUAGE, TAG_English
    if name is None:
        return TAG_English
    for tag in TAG_ALL_LANGUAGE:
        if tag.name == name:
            return tag
    return None