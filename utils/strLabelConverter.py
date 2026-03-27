import collections

import torch


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def textconvert(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        '''
        if isinstance(text, str):
            for char in enumerate(text):
                    if self._ignore_case:
                        char = char.lower()
                    if not self.dict.get(char):
                        text.replace(char,'')
            return text
        '''

        assert isinstance(text, collections.Iterable), "please set the batchsize > 1"
        textnew = []
        for s in text:
            t = []
            for char in s:
                # o不在self.dict中,alphabet中没有o,因此会走continue,需要加入下段代码,防止o被跳过
                if self._ignore_case:
                    char = char.lower()
                if char == 'i':
                    char = '1'
                elif char == 'o':
                    char = '0'
                elif char == '-':
                    char = ''
                #
                if self._ignore_case:
                    char = char.lower()
                if not self.dict.get(char):
                    t.append('*')
                    continue
                else:
                    t.append(char)
            t = ''.join(t)
            textnew.append(t)
        return textnew

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            t = []
            for char in text:
                if self._ignore_case:
                    char = char.lower()
                if char == 'i':
                    char = 1
                elif char == 'o':
                    char = 0
                elif char == '-':
                    char = ''
                    # char = '-'
                if not self.dict.get(char):
                    continue
                else:
                    t.append(self.dict[char])
            length = [len(t)]
            return (torch.IntTensor(t), torch.IntTensor(length))
        elif isinstance(text, collections.abc.Iterable):  # python >= 3.10
            # elif isinstance(text, collections.Iterable):# python <= 3.9
            length = []
            nums = []
            for s in text:
                # print(s)
                t = []
                for char in s:
                    if self._ignore_case:
                        char = char.lower()
                    if char == 'i':
                        char = '1'
                    elif char == 'o':
                        char = '0'
                        # char = 'o'
                    elif char == '-':
                        char = ''
                    if not self.dict.get(char):
                        t.append(self.dict['*'])
                        continue
                    else:
                        t.append(self.dict[char])
                length.append(len(t))
                # if len(t)!=7:
                #     print('s',s)
                #     print('t',t)
                nums.extend(t)
            # print(nums)
            return (torch.IntTensor(nums), torch.IntTensor(length))

    def encodeold(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            # print(text)
            t = []
            for char in text:
                if self._ignore_case:
                    char = char.lower()
                if not self.dict.get(char):
                    continue
                else:
                    t.append(self.dict[char])
            text = t
            # text = [
            # self.dict[char.lower() if self._ignore_case else char]
            # for char in text]
            length = [len(text)]
            # print(length)
            # print(text)
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

    def decode_with_score(self, score, t, length, raw=False):
        if length.numel() == 1:
            # 1 batch
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                score_list = []
                for i in range(length):
                    # != 占位符 && ( i==0 或者 !=上一个字符 )
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                        score_list.append(score[i])
                        # 取该字符时,对应的概率
                # print(char_list, score_list)
                score_list = torch.tensor(score_list)
                return ''.join(char_list), score_list, torch.prod(score_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            scores = []
            score_lists = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                result = self.decode_with_score(score[index:index + l], t[index:index + l], torch.IntTensor([l]),
                                                raw=raw)
                texts.append(result[0])
                score_lists.append(result[1])
                scores.append(result[2])
                index += l
            return texts, score_lists, scores