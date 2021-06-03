from ..attack_eval import AttackEval
from ..classifier import Classifier
import json, sys, time, datasets
from tqdm import tqdm
from ..utils import visualizer, result_visualizer, check_parameters
from ..exceptions import ClassifierNotSupportException
from ..text_processors import DefaultTextProcessor
import multiprocessing, logging
import numpy as np
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "processor": DefaultTextProcessor(),
    "language_tool": None,
    "language_model": None,
    "sentence_encoder": None,
    "levenshtein_tool": None,
    "modification_tool": None,
    "jaccard_word_tool": None,
    "jaccard_char_tool": None,
    "similarity_tool": None,
    "bleu_tool": None,

    "success_rate": True,
    "fluency": False,
    "mistake": False,
    "semantic": False,
    "levenstein": False,
    "word_distance": False,
    "modification_rate": False,
    "jaccard_word": False,
    "jaccard_char": False,
    "similarity": False,
    "bleu": False,
    "running_time": True,

    "num_process": 1,
}


class MetaClassifierWrapper(Classifier):
    def __init__(self, clsf):
        self.__meta = None
        self.__clsf = clsf

    def set_meta(self, meta):
        self.__meta = meta

    def get_pred(self, input_):
        return self.__clsf.get_pred(input_, self.__meta)

    def get_prob(self, input_):
        return self.__clsf.get_prob(input_, self.__meta)

    def get_grad(self, input_, labels):
        return self.__clsf.get_grad(input_, labels, self.__meta)


def worker(data):
    attacker = globals()["$WORKER_ATTACKER"]
    classifier = globals()["$WORKER_CLASSIFIER"]

    clsf_wrapper = MetaClassifierWrapper(classifier)
    clsf_wrapper.set_meta(data)
    if "target" in data:
        res = attacker(classifier, data["x"], data["target"])
    else:
        res = attacker(classifier, data["x"])
    return data, res


def worker_init(attacker, classifier):
    globals()['$WORKER_ATTACKER'] = attacker
    globals()['$WORKER_CLASSIFIER'] = classifier


class DetailedAttackEval(AttackEval):
    """
    DefaultAttackEval is the default implementation of AttackEval that provides basic evaluation functions.

    In this class, there are four key methods that maybe useful for your extension.

    * **measure:** Measures the basic metrics.
    * **update:** Accumulates the measurement.
    * **get_result:** Calculates the final results.
    * **clear:** Clear all the Accumulated results.

    The workflow is: ``clear -> measure -> update -> measure -> update -> ... -> get_result``.
    You can override these four methods to add your custom measurement.

    See :doc:`Example 4 </examples/example4>` for detail.

    """

    def __init__(self, attacker, classifier, **kwargs):
        """
        :param Attacker attacker: The attacker you use.
        :param Classifier classifier: The classifier you want to attack.
        :param bool running_time: If true, returns "Avg. Running Time" in summary. **Default:** True
        :param bool success_rate: If true, returns "Attack Success Rate". **Default:** True
        :param bool fluency: If true, returns "Avg. Fluency (ppl)". **Default:** False
        :param bool mistake: If true, returns "Avg. Grammatical Errors". **Default:** False
        :param bool semantic: If true, returns "Avg. Semantic Similarity". **Default:** False
        :param bool levenstein: If true, returns "Avg. Levenshtein Edit Distance". **Default:** False
        :param bool word_distance: If true, applies token-level levenstein edit distance. **Default:** False
        :param bool modification_rate: If true, returns "Avg. Word Modif. Rate". **Default:** False
        :param TextProcessor processor: Text processor used in DefaultAttackEval. **Default:** :any:`DefaultTextProcessor`
        :param int num_process: The number of parallel threads at the time of the evaluation. **Default:** 1

        :Package Requirements:
            * **language_tool_python** (for `mistake` option)
            * **Java** (for `mistake` option)
            * **transformers** (for `fluency` option)
            * **tensorflow** >= 2.0.0 (for `semantic` option)
            * **tensorflow_hub** (for `semantic` option)

        :Data Requirements:
            * :py:data:`.UniversalSentenceEncoder` (for `semantic` option)
        """
        self.__config = DEFAULT_CONFIG.copy()
        self.__config.update(kwargs)
        check_parameters(DEFAULT_CONFIG.keys(), self.__config)
        self.clear()
        self.attacker = attacker
        self.classifier = classifier

        if self.__config["levenstein"]:
            if self.__config["levenshtein_tool"] is None:
                from ..metric import Levenshtein
                self.__config["levenshtein_tool"] = Levenshtein()

        if self.__config["mistake"]:
            if self.__config["language_tool"] is None:
                from ..metric import LanguageTool
                self.__config["language_tool"] = LanguageTool()

        if self.__config["fluency"]:
            if self.__config["language_model"] is None:
                from ..metric import GPT2LM
                self.__config["language_model"] = GPT2LM()

        if self.__config["semantic"]:
            if self.__config["sentence_encoder"] is None:
                from ..metric import UniversalSentenceEncoder
                self.__config["sentence_encoder"] = UniversalSentenceEncoder()

        if self.__config["modification_rate"]:
            if self.__config["modification_tool"] is None:
                from ..metric import Modification
                self.__config["modification_tool"] = Modification()

        if self.__config["jaccard_word"]:
            if self.__config["jaccard_word_tool"] is None:
                from ..metric import Jaccard_Word
                self.__config["jaccard_word_tool"] = Jaccard_Word()

        if self.__config["jaccard_char"]:
            if self.__config["jaccard_char_tool"] is None:
                from ..metric import Jaccard_Char
                self.__config["jaccard_char_tool"] = Jaccard_Char()

        if self.__config["similarity"]:
            if self.__config["similarity_tool"] is None:
                from ..metric import Sim_Cos
                self.__config["similarity_tool"] = Sim_Cos()

        if self.__config["bleu"]:
            if self.__config["bleu_tool"] is None:
                from ..metric import BLEU
                self.__config["bleu_tool"] = BLEU()

        if self.__config["num_process"] > 1:
            if multiprocessing.get_start_method() != "spawn":
                logger.warning(
                    "Warning: multiprocessing start method '%s' may cause pytorch.cuda initialization error.",
                    multiprocessing.get_start_method())

    def eval(self, dataset, total_len=None, visualize=False, progress_bar=False):
        """
        :param Datasets.Dataset dataset: A :py:class:`Datasets.Dataset`.
        :param int total_len: If `dataset` is a generator, total_len is passed the progress bar.
        :param bool visualize: Display a visualized result for each instance and the summary.
        :param bool progress_bar: Dispaly a progress bar(tqdm). **Default:** True
        :return: Returns a dict of the summary.
        :rtype: dict

        In this method, ``eval_results`` is called and gets the result for each instance iteratively.
        """
        if hasattr(dataset, "__len__"):
            total_len = len(dataset)

        counter = 0

        def tqdm_writer(x):
            return tqdm.write(x, end="")

        if self.__config["num_process"] > 1:
            with self.__get_pool() as pool:
                time_start = time.time()
                res_info = []
                for data, x_adv, y_adv, info in (
                tqdm(self.eval_results(dataset, pool), total=total_len) if progress_bar else self.eval_results(
                        dataset)):
                    x_orig = data["x"]
                    counter += 1
                    if info["Succeed"] == False:
                        y_orig = self.classifier.get_prob([x_orig], data)[0]
                        label_orig = np.argmax(y_orig)
                        res_info.append(
                            {"Succeed": False, 'Orig': x_orig, 'Orig Label': label_orig, 'Orig Confidence': y_orig})
                    else:
                        res = self.classifier.get_prob([x_orig, x_adv], data)
                        y_orig = res[0]
                        y_adv = res[1]
                        label_orig = np.argmax(y_orig)
                        label_adv = np.argmax(y_adv)
                        res_info.append(
                            {'Succeed': True, 'Orig': x_orig, 'Orig Label': label_orig, 'Orig Confidence': y_orig,
                             'Adv': x_adv, 'Adv Label': label_adv, 'Adv Confidence': y_adv, 'info': info})
                    if visualize:
                        try:
                            if x_adv is not None:
                                res = self.classifier.get_prob([x_orig, x_adv], data)
                                y_orig = res[0]
                                y_adv = res[1]
                            else:
                                y_orig = self.classifier.get_prob([x_orig], data)[0]
                        except ClassifierNotSupportException:
                            if x_adv is not None:
                                res = self.classifier.get_pred([x_orig, x_adv], data)
                                y_orig = int(res[0])
                                y_adv = int(res[1])
                            else:
                                y_orig = int(self.classifier.get_pred([x_orig], data)[0])

                        if progress_bar:
                            visualizer(counter, x_orig, y_orig, x_adv, y_adv, info, tqdm_writer, self.__get_tokens)
                        else:
                            visualizer(counter, x_orig, y_orig, x_adv, y_adv, info, sys.stdout.write, self.__get_tokens)
                res = self.get_result()
                if self.__config["running_time"]:
                    res["Avg. Running Time"] = (time.time() - time_start) / counter
        else:
            time_start = time.time()
            res_info=[]
            for data, x_adv, y_adv, info in (
            tqdm(self.eval_results(dataset), total=total_len) if progress_bar else self.eval_results(dataset)):
                x_orig = data["x"]

                counter += 1
                if info["Succeed"]==False:
                    y_orig = self.classifier.get_prob([x_orig], data)[0]
                    label_orig = np.argmax(y_orig)
                    res_info.append({"Succeed":False,'Orig':x_orig,'Orig Label':label_orig,'Orig Confidence':y_orig})
                else:
                    res = self.classifier.get_prob([x_orig, x_adv], data)
                    y_orig = res[0]
                    y_adv = res[1]
                    label_orig=np.argmax(y_orig)
                    label_adv=np.argmax(y_adv)
                    res_info.append({'Succeed':True,'Orig':x_orig,'Orig Label':label_orig,'Orig Confidence':y_orig,'Adv':x_adv,'Adv Label':label_adv,'Adv Confidence':y_adv,'info':info})
                if visualize:
                    try:
                        if x_adv is not None:
                            res = self.classifier.get_prob([x_orig, x_adv], data)
                            y_orig = res[0]
                            y_adv = res[1]
                        else:
                            y_orig = self.classifier.get_prob([x_orig], data)[0]
                    except ClassifierNotSupportException:
                        if x_adv is not None:
                            res = self.classifier.get_pred([x_orig, x_adv], data)
                            y_orig = int(res[0])
                            y_adv = int(res[1])
                        else:
                            y_orig = int(self.classifier.get_pred([x_orig], data)[0])
                    if progress_bar:
                        visualizer(counter, x_orig, y_orig, x_adv, y_adv, info, tqdm_writer, self.__get_tokens)
                    else:
                        visualizer(counter, x_orig, y_orig, x_adv, y_adv, info, sys.stdout.write, self.__get_tokens)

            res = self.get_result()
            if self.__config["running_time"]:
                res["Avg. Running Time"] = (time.time() - time_start) / counter

        if visualize:
            result_visualizer(res, sys.stdout.write)
        return res,res_info

    def __get_pool(self):
        return multiprocessing.Pool(self.__config["num_process"], initializer=worker_init,
                                    initargs=(self.attacker, self.classifier))

    def print(self):
        print(json.dumps(self.get_result(), indent="\t"))

    def dump(self, file_like_object):
        json.dump(self.get_result(), file_like_object)

    def dumps(self):
        return json.dumps(self.get_result())

    def __update(self, sentA, sentB):
        info = self.measure(sentA, sentB)
        return self.update(info)

    def eval_results(self, dataset, __pool=None):
        """
        :param Datasets.Dataset dataset: A :py:class:`Datasets.Dataset`.
        :return: A generator which generates the result for each instance, *(data, x_adv, y_adv, info)*.
        :rtype: generator
        """
        self.clear()

        def _iter_gen():
            for data in dataset:
                yield data

        if self.__config["num_process"] > 1:
            if __pool is None:
                with self.__get_pool() as pool:
                    for data, res in pool.imap(worker, _iter_gen(), chunksize=self.__config["num_process"] * 2):
                        if res is None:
                            info = self.__update(data["x"], None)
                        else:
                            info = self.__update(data["x"], res[0])
                        if not info["Succeed"]:
                            yield (data, None, None, info)
                        else:
                            yield (data, res[0], res[1], info)
            else:
                for data, res in __pool.imap(worker, _iter_gen(), chunksize=self.__config["num_process"] * 2):
                    if res is None:
                        info = self.__update(data["x"], None)
                    else:
                        info = self.__update(data["x"], res[0])
                    if not info["Succeed"]:
                        yield (data, None, None, info)
                    else:
                        yield (data, res[0], res[1], info)
        else:
            clsf_wrapper = MetaClassifierWrapper(self.classifier)
            for data in _iter_gen():
                # assert isinstance(data, DataInstance)
                clsf_wrapper.set_meta(data)
                if "target" in data:
                    res = self.attacker(self.classifier, data["x"], data["target"])
                else:
                    res = self.attacker(self.classifier, data["x"])
                if res is None:
                    info = self.__update(data["x"], None)
                else:
                    info = self.__update(data["x"], res[0])
                if not info["Succeed"]:
                    yield (data, None, None, info)
                else:
                    yield (data, res[0], res[1], info)

    def __levenshtein(self, sentA, sentB):
        return self.__config["levenshtein_tool"](sentA, sentB)

    def __get_tokens(self, sent):
        return list(map(lambda x: x[0], self.__config["processor"].get_tokens(sent)))

    def __get_mistakes(self, sent):
        return self.__config["language_tool"](sent)

    def __get_fluency(self, sent):
        if len(sent.strip()) == 0:
            return 1
        return self.__config["language_model"](sent)

    def __get_semantic(self, sentA, sentB):
        return self.__config["sentence_encoder"](sentA, sentB)

    def __get_modification(self, sentA, sentB):
        tokenA = self.__get_tokens(sentA)
        tokenB = self.__get_tokens(sentB)
        return self.__config["modification_tool"](tokenA, tokenB)

    def __get_jaccard_word(self, sentA, sentB):
        tokenA = self.__get_tokens(sentA)
        tokenB = self.__get_tokens(sentB)
        return self.__config["jaccard_word_tool"](tokenA, tokenB)

    def __get_jaccard_char(self, sentA, sentB):
        return self.__config["jaccard_char_tool"](sentA, sentB)

    def __get_similarity(self, sentA, sentB):
        return self.__config["similarity_tool"](sentA, sentB)

    def __get_bleu(self, sentA, sentB):
        tokenA = self.__get_tokens(sentA)
        tokenB = self.__get_tokens(sentB)
        return self.__config["bleu_tool"](tokenA, tokenB)

    def measure(self, input_, attack_result):
        """
        :param str input_: The original sentence.
        :param attack_result: The adversarial sentence which is generated by attackers. If is None, means attacker failed to generate an adversarial sentence.
        :type attack_result: str or None
        :return: A dict contains all the results for this instance.
        :rtype: dict

        In this method, we measure all the metrics which corresponding options are setted to True.
        """
        if attack_result is None:
            return {"Succeed": False}

        info = {"Succeed": True}

        if self.__config["levenstein"]:
            va = input_
            vb = attack_result
            if self.__config["word_distance"]:
                va = self.__get_tokens(va)
                vb = self.__get_tokens(vb)
            info["Edit Distance"] = self.__levenshtein(va, vb)

        if self.__config["mistake"]:
            info["Grammatical Errors"] = self.__get_mistakes(attack_result)

        if self.__config["fluency"]:
            info["Fluency (ppl)"] = self.__get_fluency(attack_result)

        if self.__config["semantic"]:
            info["Semantic Similarity"] = self.__get_semantic(input_, attack_result)

        if self.__config["modification_rate"]:
            info["Word Modif. Rate"] = self.__get_modification(input_, attack_result)

        if self.__config["jaccard_word"]:
            info["Jaccard Word Similarity"] = self.__get_jaccard_word(input_, attack_result)

        if self.__config["jaccard_char"]:
            info["Jaccard Char Similarity"] = self.__get_jaccard_char(input_, attack_result)

        if self.__config["similarity"]:
            info["Sentence Similarity"] = self.__get_similarity(input_, attack_result)

        if self.__config["bleu"]:
            info["BLEU"] = self.__get_bleu(input_, attack_result)
        return info

    def update(self, info):
        """
        :param dict info: The result returned by ``measure`` method.
        :return: Just return the parameter **info**.
        :rtype: dict

        In this method, we accumulate the results from ``measure`` method.
        """
        if "total" not in self.__result:
            self.__result["total"] = 0
        self.__result["total"] += 1

        if self.__config["success_rate"]:
            if "succeed" not in self.__result:
                self.__result["succeed"] = 0
            if info["Succeed"]:
                self.__result["succeed"] += 1

        # early stop
        if not info["Succeed"]:
            return info

        if self.__config["levenstein"]:
            if "edit" not in self.__result:
                self.__result["edit"] = 0
            self.__result["edit"] += info["Edit Distance"]

        if self.__config["mistake"]:
            if "mistake" not in self.__result:
                self.__result["mistake"] = 0
            self.__result["mistake"] += info["Grammatical Errors"]

        if self.__config["fluency"]:
            if "fluency" not in self.__result:
                self.__result["fluency"] = 0
            self.__result["fluency"] += info["Fluency (ppl)"]

        if self.__config["semantic"]:
            if "semantic" not in self.__result:
                self.__result["semantic"] = 0
            self.__result["semantic"] += info["Semantic Similarity"]

        if self.__config["modification_rate"]:
            if "modification" not in self.__result:
                self.__result["modification"] = 0
            self.__result["modification"] += info["Word Modif. Rate"]

        if self.__config["jaccard_word"]:
            if "jaccard_word" not in self.__result:
                self.__result["jaccard_word"] = 0
            self.__result["jaccard_word"] += info["Jaccard Word Similarity"]

        if self.__config["jaccard_char"]:
            if "jaccard_char" not in self.__result:
                self.__result["jaccard_char"] = 0
            self.__result["jaccard_char"] += info["Jaccard Char Similarity"]

        if self.__config["similarity"]:
            if "similarity" not in self.__result:
                self.__result["similarity"] = 0
            self.__result["similarity"] += info["Sentence Similarity"]

        if self.__config["bleu"]:
            if "bleu" not in self.__result:
                self.__result["bleu"] = 0
            self.__result["bleu"] += info["BLEU"]
        return info

    def get_result(self):
        """
        :return: The results which were accumulated previously.
        :rtype: dict

        This method summarizes and returns to previous accumulated results.
        """
        ret = {}
        ret["Total Attacked Instances"] = self.__result["total"]
        if self.__config["success_rate"]:
            ret["Successful Instances"] = self.__result["succeed"]
            ret["Attack Success Rate"] = self.__result["succeed"] / self.__result["total"]
        if self.__result["succeed"] > 0:
            if self.__config["levenstein"]:
                if "edit" not in self.__result:
                    self.__result["edit"] = 0
                ret["Avg. Levenshtein Edit Distance"] = self.__result["edit"] / self.__result["succeed"]
            if self.__config["mistake"]:
                if "mistake" not in self.__result:
                    self.__result["mistake"] = 0
                ret["Avg. Grammatical Errors"] = self.__result["mistake"] / self.__result["succeed"]
            if self.__config["fluency"]:
                if "fluency" not in self.__result:
                    self.__result["fluency"] = 0
                ret["Avg. Fluency (ppl)"] = self.__result["fluency"] / self.__result["succeed"]
            if self.__config["semantic"]:
                if "semantic" not in self.__result:
                    self.__result["semantic"] = 0
                ret["Avg. Semantic Similarity"] = self.__result["semantic"] / self.__result["succeed"]
            if self.__config["modification_rate"]:
                if "modification" not in self.__result:
                    self.__result["modification"] = 0
                ret["Avg. Word Modif. Rate"] = self.__result["modification"] / self.__result["succeed"]

            if self.__config["jaccard_word"]:
                if "jaccard_word" not in self.__result:
                    self.__result["jaccard_word"] = 0
                ret["Avg. Jaccard Word Similarity"] = self.__result["jaccard_word"] / self.__result["succeed"]

            if self.__config["jaccard_char"]:
                if "jaccard_char" not in self.__result:
                    self.__result["jaccard_char"] = 0
                ret["Avg. Jaccard Char Similarity"] = self.__result["jaccard_char"] / self.__result["succeed"]

            if self.__config["similarity"]:
                if "similarity" not in self.__result:
                    self.__result["similarity"] = 0
                ret["Avg. Sentence Similarity"] = self.__result["similarity"] / self.__result["succeed"]

            if self.__config["bleu"]:
                if "bleu" not in self.__result:
                    self.__result["bleu"] = 0
                ret["Avg. BLEU"] = self.__result["bleu"] / self.__result["succeed"]
        return ret

    def clear(self):
        """
        Clear all the accumulated results.
        """
        self.__result = {}

    def generate_adv(self, dataset, total_len=None):
        """
        :param Datasets.Dataset dataset: A :py:class:`Datasets.Dataset`.
        :return: A :py:class:`Datasets.Dataset` consists of adversarial samples.
        :rtype: datasets.Dataset
        """
        if hasattr(dataset, "__len__"):
            total_len = len(dataset)

        ret = {"x": [], "y": [], "pred": [], "original": [], "info": []}
        for data, x_adv, y_adv, info in (
        tqdm(self.eval_results(dataset), total=total_len) if self.__progress_bar else self.eval_results(dataset)):
            if x_adv is not None:
                ret["x"].append(x_adv)
                ret["y"].append(data["y"])
                ret["pred"].append(y_adv)
                ret["original"].append(data["x"])
                ret["info"].append(info)
        return datasets.Dataset.from_dict(ret)

