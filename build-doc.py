import os
import OpenAttack

def getSubClasses(module, clss):
    ret = []
    for name in dir(module):
        try:
            if issubclass(module.__dict__[name], clss):
                ret.append(name)
        except TypeError:
            continue
    return ret

def getDocMembers(clss):
    ret = []
    for kw in dir(clss):
        if kw.startswith("_"):
            continue
        if clss.__dict__[kw].__doc__ is not None:
            ret.append(kw)
    return ret

def make_attacker(path):
    addition_members = {
        "UATAttacker": ["get_triggers"],
    }
    opt = "===================\nAttackers API\n===================\n\n"

    opt += "Attacker\n=============\n\n.. autoclass:: OpenAttack.Attacker\n    :members:\n\n"
    opt += "-" * 36 + "\n\n"
    
    opt += "ClassificationAttacker\n==============================\n\n.. autoclass:: OpenAttack.attackers.ClassificationAttacker\n    :members:\n\n"
    opt += "-" * 36 + "\n\n"
    
    
    for name in getSubClasses(OpenAttack.attackers, OpenAttack.ClassificationAttacker):
        if name == "ClassificationAttacker":
            continue
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.attackers.%s(OpenAttack.attackers.ClassificationAttacker)\n" % name

        members = ["__init__"] + (addition_members[name] if name in addition_members else [])
        opt += "    :members: " + (", ".join(members)) + "\n\n"
    open(path, "w", encoding="utf-8").write(opt)
    return opt

def make_attack_eval(path):
    opt = "========================\nAttackEvals API\n========================\n\n"
    opt += "AttackEval\n----------------\n\n.. autoclass:: OpenAttack.AttackEval\n    :members: __init__, eval, ieval\n\n"
    open(path, "w", encoding="utf-8").write(opt)

def make_victim(path):
    addition_members = {
        "TransformersClassifier": ["to"],
    }
    skip_list = {"Classifier"}

    opt = "===================\nVictims API\n===================\n\n"
    opt += "Classifier\n===========================\n\n.. autoclass:: OpenAttack.victim.classifiers.Classifier\n    :members:\n\n"
    opt += "-" * 36 + "\n\n"
    
    for name in getSubClasses(OpenAttack.victim.classifiers, OpenAttack.victim.classifiers.Classifier):
        if name in skip_list:
            continue

        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.classifiers.%s(OpenAttack.Classifier)\n" % name

        members = ["__init__"] + (addition_members[name] if name in addition_members else [])
        opt += "    :members: " + (", ".join(members)) + "\n\n"
    open(path, "w", encoding="utf-8").write(opt)
    return opt
    
def make_data_manager(path):
    opt = "===================\nDataManager API\n===================\n\n.. autoclass:: OpenAttack.DataManager\n    :members:"
    open(path, "w", encoding="utf-8").write(opt)
    return opt

def make_data(path):
    import pkgutil
    cats = {

    }
    for data in pkgutil.iter_modules(OpenAttack.data.__path__):
        data = data.module_finder.find_loader(data.name)[0].load_module()
        if hasattr(data, "NAME") and (data.NAME in OpenAttack.DataManager.AVAILABLE_DATAS):
            name = data.NAME
            if name == "test":
                continue
            cat = name.split(".")
            if len(cat) == 1:
                continue
            name = ".".join(cat[1:])
            cat = cat[0]
            pack = data.__name__

            if cat not in cats:
                cats[cat] = []
            cats[cat].append({
                "name": name,
                "package":  pack
            })
    

    
    for cat in cats.keys():
        opt = "=====================\n%s\n=====================\n\n" % cat
        opt += ".. _label-data-%s:\n\n" % cat
        for data in cats[cat]:
            opt += data["name"] + "\n" + ("-" * (2 + len(data["name"]))) + "\n\n"
            opt += ".. py:data:: " + cat + "." + data["name"] + "\n\n"
            opt += "    .. automodule:: OpenAttack.data." + data["package"] + "\n\n"
        open(os.path.join(path, cat + ".rst"), "w", encoding="utf-8").write(opt)
    return opt

def make_metric(path):
    opt = "==================\nMetric API\n==================\n\n"
    

    metrics = []
    selector = []
    for name in dir(OpenAttack.metric):
        if isinstance(OpenAttack.metric.__dict__[name], type):
            if issubclass(OpenAttack.metric.__dict__[name], OpenAttack.metric.AttackMetric):
                if name == "AttackMetric":
                    continue
                metrics.append(name)
            elif issubclass(OpenAttack.metric.__dict__[name], OpenAttack.metric.MetricSelector):
                if name == "MetricSelector":
                    continue
                selector.append(name)
    
    opt += """Attacker Metrics
==================

.. autoclass:: OpenAttack.metric.AttackMetric
    :members:


"""

    for name in metrics:
        cls = OpenAttack.metric.__dict__[name]
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.metric." + name + "\n"
        if hasattr(cls, "calc_score"):
            opt += "    :members: __init__, calc_score" + "\n"
        else:
            opt += "    :members: __init__" + "\n"
        opt += "    :exclude-members: TAGS" + "\n\n"
    
    opt += """Metrics Selector
=======================

.. autoclass:: OpenAttack.metric.MetricSelector
    :members:


"""
    
    for name in selector:
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.metric." + name + "\n"
        opt += "    :members: " + "\n"
        opt += "    :exclude-members: TAGS" + "\n\n"
    
    open(path, "w", encoding="utf-8").write(opt)
    return opt

def make_substitute(path):
    opt = "======================\nSubstitutes API\n======================\n\n"
    opt += """

Abstract Classes
------------------------

.. autoclass:: OpenAttack.attack_assist.substitute.word.WordSubstitute
    :members: __call__

.. autoclass:: OpenAttack.attack_assist.substitute.char.CharSubstitute
    :members: __call__

-------------------------------------------------------------------------------


"""
    subs = getSubClasses(OpenAttack.attack_assist.substitute.word, OpenAttack.attack_assist.substitute.word.WordSubstitute)
    embed_based_idx = subs.index("EmbedBasedSubstitute")
    if embed_based_idx != -1:
        subs[0], subs[embed_based_idx] = subs[embed_based_idx], subs[0]
    
    for name in subs:
        cls = OpenAttack.attack_assist.substitute.word.__dict__[name]
        if cls is OpenAttack.attack_assist.substitute.word.WordSubstitute:
            continue
        
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.attack_assist.substitute.word.%s(OpenAttack.attack_assist.substitute.word.WordSubstitute)\n" % name
        opt += "    :members: __init__\n\n"

    subs = getSubClasses(OpenAttack.attack_assist.substitute.char, OpenAttack.attack_assist.substitute.char.CharSubstitute)
    
    for name in subs:
        cls = OpenAttack.attack_assist.substitute.char.__dict__[name]
        if cls is OpenAttack.attack_assist.substitute.char.CharSubstitute:
            continue
        
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.attack_assist.substitute.char.%s(OpenAttack.attack_assist.substitute.char.CharSubstitute)\n" % name
        opt += "    :members: __init__\n\n"

    open(path, "w", encoding="utf-8").write(opt)
    return opt

def make_text_processor(path):
    opt = "========================\nText Processors API\n========================\n\n"
    opt += "Tokenizers\n============================\n\n"
    opt += """
.. autoclass:: OpenAttack.text_process.tokenizer.Tokenizer
    :members: tokenize, detokenize

"""

    import OpenAttack.text_process.tokenizer
    
    for name in getSubClasses(OpenAttack.text_process.tokenizer, OpenAttack.text_process.tokenizer.Tokenizer):
        if name == "Tokenizer":
            continue
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.text_process.tokenizer.%s(OpenAttack.text_process.tokenizer.Tokenizer)\n" % name
        opt += "    :members:\n\n"
    
    
    import OpenAttack.text_process.lemmatizer
    
    opt += "Lemmatizer\n============================\n\n"
    opt += """
.. autoclass:: OpenAttack.text_process.lemmatizer.Lemmatizer
    :members: lemmatize, delemmatize

"""
    for name in getSubClasses(OpenAttack.text_process.lemmatizer, OpenAttack.text_process.lemmatizer.Lemmatizer):
        if name == "Lemmatizer":
            continue
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.text_process.lemmatizer.%s(OpenAttack.text_process.lemmatizer.Lemmatizer)\n" % name
        opt += "    :members:\n\n"

    
    import OpenAttack.text_process.constituency_parser
    
    opt += "ConstituencyParser\n============================\n\n"
    opt += """
.. autoclass:: OpenAttack.text_process.constituency_parser.ConstituencyParser
    :members: __call__

"""
    for name in getSubClasses(OpenAttack.text_process.constituency_parser, OpenAttack.text_process.constituency_parser.ConstituencyParser):
        if name == "ConstituencyParser":
            continue
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.text_process.constituency_parser.%s(OpenAttack.text_process.constituency_parser.ConstituencyParser)\n" % name
        opt += "    :members:\n\n"
    

    open(path, "w", encoding="utf-8").write(opt)
    return opt

def make_utils(path):
    opt = "=====================\nutils API\n=====================\n\n"
    for name in OpenAttack.utils.__dir__():
        if name.startswith("__"):
            continue
        obj = OpenAttack.utils.__dict__[name]
        if type(obj).__name__ == "module":
            continue
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        if type(obj).__name__  == "function":
            opt += ".. autofunction:: OpenAttack.utils." + name + "\n\n"
        else:
            opt += ".. autoclass:: OpenAttack.utils." + name + "\n"
            opt += "    :members: " + "\n\n"
    open(path, "w", encoding="utf-8").write(opt)
    return opt

def main(path):
    make_attacker(os.path.join(path, "attacker.rst"))
    make_attack_eval(os.path.join(path, "attack_eval.rst"))
    make_victim(os.path.join(path, "victim.rst"))
    make_data_manager(os.path.join(path, "data_manager.rst"))
    make_data(os.path.join(path, "..", "data"))
    make_metric(os.path.join(path, "metric.rst"))
    make_substitute(os.path.join(path, "substitute.rst"))
    make_text_processor(os.path.join(path, "text_processor.rst"))
    make_utils(os.path.join(path, "utils.rst"))

if __name__ == "__main__":
    import sys
    path = os.path.abspath(sys.argv[1])
    main(path)