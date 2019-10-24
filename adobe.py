#!/usr/bin/env python

import os
import tqdm
import glob
import json
import ember
import pefile
import numpy as np
import pandas as pd
import lightgbm as lgb
import multiprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, make_scorer)


class AdobeEval:

    ordered_features = [
        "DebugSize", "ImageVersion", "IatRVA", "ExportSize", "ResourceSize", "VirtualSize2", "NumberOfSections"
    ]
    dim = len(ordered_features)

    def __init__(self, path=None, raw_features=None):
        if path is not None and raw_features is not None:
            raise Exception("Cannot initialize with both a path and raw features.")

        if path is not None:
            self.from_path(path)
        elif raw_features is not None:
            self.from_raw_features(raw_features)

    def from_raw_features(self, raw_features):
        try:
            self.DebugSize = raw_features["datadirectories"][6]["size"]
            self.ImageVersion = ((raw_features["header"]["optional"]["major_image_version"] * 100 +
                                  raw_features["header"]["optional"]["minor_image_version"]) * 1000)
            self.IatRVA = raw_features["datadirectories"][1]["virtual_address"]
            self.ExportSize = raw_features["datadirectories"][0]["size"]
            self.ResourceSize = raw_features["datadirectories"][2]["size"]
            self.NumberOfSections = len(raw_features["section"]["sections"])
            if len(raw_features["section"]["sections"]) < 2:
                self.VirtualSize2 = 0
            else:
                self.VirtualSize2 = raw_features["section"]["sections"][1]["vsize"]
            self.init_success = True
        except Exception:
            self.init_success = False

    def from_path(self, path):
        try:
            pef = pefile.PE(path, fast_load=True)
            self.DebugSize = pef.OPTIONAL_HEADER.DATA_DIRECTORY[6].Size
            self.ImageVersion = (
                (pef.OPTIONAL_HEADER.MajorImageVersion * 100) + pef.OPTIONAL_HEADER.MinorImageVersion) * 1000
            self.IatRVA = pef.OPTIONAL_HEADER.DATA_DIRECTORY[1].VirtualAddress
            self.ExportSize = pef.OPTIONAL_HEADER.DATA_DIRECTORY[0].Size
            self.ResourceSize = pef.OPTIONAL_HEADER.DATA_DIRECTORY[2].Size
            self.NumberOfSections = pef.FILE_HEADER.NumberOfSections
            if len(pef.sections) < 2:
                self.VirtualSize2 = 0
            else:
                self.VirtualSize2 = pef.sections[1].Misc_VirtualSize
            self.init_success = True
        except Exception:
            self.init_success = False

    def feature_vector(self):
        if self.init_success:
            return np.array([self.__dict__[f] for f in self.ordered_features], dtype=np.float32)
        else:
            return np.zeros(len(self.ordered_features), dtype=np.float32)

    def data_dump(self):
        for f in self.ordered_features:
            print(f"{f+':':<18}{self.__dict__[f]}")

    def __eq__(self, other):
        return (self.DebugSize == other.DebugSize and self.ImageVersion == other.ImageVersion and
                self.IatRVA == other.IatRVA and self.ExportSize == other.ExportSize and
                self.ResourceSize == other.ResourceSize and self.VirtualSize2 == other.VirtualSize2 and
                self.NumberOfSections == other.NumberOfSections)

    def runJ48(self):
        isDirty = 0
        if self.DebugSize <= 0:
            if self.ExportSize <= 211:
                if self.ImageVersion <= 520:
                    if self.VirtualSize2 <= 130:
                        if self.VirtualSize2 <= 5:
                            if self.ResourceSize <= 37520:
                                isDirty = 1
                            elif self.ResourceSize > 37520:
                                if self.NumberOfSections <= 2:
                                    if self.IatRVA <= 2048:
                                        isDirty = 0
                                    else:
                                        isDirty = 1
                                else:
                                    isDirty = 1
                        else:
                            if self.VirtualSize2 <= 12:
                                if self.NumberOfSections <= 3:
                                    isDirty = 0
                                else:
                                    isDirty = 1
                            else:
                                isDirty = 1
                    else:
                        isDirty = 1
                else:
                    if self.ResourceSize <= 0:
                        if self.ImageVersion <= 1000:
                            if self.NumberOfSections <= 4:
                                isDirty = 1
                            else:
                                if self.ExportSize <= 74:
                                    if self.VirtualSize2 <= 1556:
                                        isDirty = 1
                                    else:
                                        isDirty = 0
                                else:
                                    isDirty = 0
                        else:
                            isDirty = 1
                    else:
                        if self.NumberOfSections <= 2:
                            if self.ImageVersion <= 3420:
                                isDirty = 1
                            else:
                                isDirty = 0
                        else:
                            isDirty = 1
            else:
                if self.ImageVersion <= 0:
                    if self.ExportSize <= 23330:
                        if self.IatRVA <= 98304:
                            if self.NumberOfSections <= 3:
                                isDirty = 1
                            else:
                                if self.IatRVA <= 53872:
                                    isDirty = 0
                                else:
                                    if self.ExportSize <= 273:
                                        isDirty = 1
                                    else:
                                        if self.ResourceSize <= 1016:
                                            isDirty = 1
                                        else:
                                            isDirty = 0
                        else:
                            isDirty = 0
                    else:
                        isDirty = 1
                else:
                    isDirty = 0
        else:
            if self.ResourceSize <= 545:
                if self.ExportSize <= 92:
                    if self.NumberOfSections <= 4:
                        isDirty = 0
                    else:
                        isDirty = 1
            else:
                if self.IatRVA <= 94208:
                    if self.NumberOfSections <= 5:
                        if self.ExportSize <= 0:
                            if self.NumberOfSections <= 4:
                                if self.IatRVA <= 13504:
                                    if self.ImageVersion <= 353:
                                        if self.NumberOfSections <= 3:
                                            if self.IatRVA <= 6144:
                                                if self.IatRVA <= 2048:
                                                    isDirty = 0
                                                else:
                                                    if self.VirtualSize2 <= 496:
                                                        isDirty = 1
                                                    else:
                                                        isDirty = 0
                                            else:
                                                isDirty = 0
                                        else:
                                            if self.DebugSize <= 41:
                                                if self.ResourceSize <= 22720:
                                                    isDirty = 1
                                                else:
                                                    isDirty = 0
                                            else:
                                                isDirty = 0
                                    else:
                                        isDirty = 0
                                else:
                                    if self.ResourceSize <= 35328:
                                        isDirty = 0
                                    else:
                                        isDirty = 1
                            else:
                                if self.IatRVA <= 2048:
                                    isDirty = 1
                                else:
                                    isDirty = 0
                        else:
                            isDirty = 0
                    else:
                        if self.IatRVA <= 1054:
                            if self.ExportSize <= 218:
                                if self.IatRVA <= 704:
                                    isDirty = 1
                                else:
                                    if self.NumberOfSections <= 6:
                                        isDirty = 1
                                    else:
                                        isDirty = 0
                            else:
                                isDirty = 0
                        else:
                            isDirty = 0
                else:
                    if self.ExportSize <= 0:
                        if self.VirtualSize2 <= 78800:
                            if self.NumberOfSections <= 4:
                                isDirty = 0
                            else:
                                if self.ImageVersion <= 2340:
                                    if self.ResourceSize <= 7328:
                                        isDirty = 1
                                    else:
                                        isDirty = 0
                                else:
                                    isDirty = 0
                        else:
                            isDirty = 1
                    else:
                        if self.IatRVA <= 106496:
                            if self.ResourceSize <= 2800:
                                isDirty = 0
                            else:
                                isDirty = 1
                        else:
                            isDirty = 0
        return isDirty

    def runJ48Graft(self):
        isDirty = 0
        if self.DebugSize <= 0:
            if self.ExportSize <= 211:
                if self.ImageVersion <= 520:
                    if self.VirtualSize2 <= 130:
                        if self.VirtualSize2 <= 5:
                            if self.ResourceSize <= 37520:
                                isDirty = 1
                            elif self.ResourceSize > 37520:
                                if self.NumberOfSections <= 2:
                                    if self.IatRVA <= 2048:
                                        if self.ExportSize <= 67.5:
                                            isDirty = 0
                                        else:
                                            isDirty = 1
                                    else:
                                        isDirty = 1
                                else:
                                    isDirty = 1
                        else:
                            if self.VirtualSize2 <= 12:
                                if self.NumberOfSections <= 3:
                                    isDirty = 0
                                else:
                                    isDirty = 1
                            else:
                                isDirty = 1
                    else:
                        isDirty = 1
                else:
                    if self.ResourceSize <= 0:
                        if self.ImageVersion <= 1000:
                            if self.NumberOfSections <= 4:
                                isDirty = 1
                            else:
                                if self.ExportSize <= 74:
                                    if self.VirtualSize2 <= 1556:
                                        isDirty = 1
                                    else:
                                        if self.IatRVA <= 5440:
                                            if self.VirtualSize2 <= 126474:
                                                if self.ExportSize <= 24:
                                                    isDirty = 0
                                                else:
                                                    isDirty = 1
                                            else:
                                                isDirty = 1
                                        else:
                                            isDirty = 1
                                else:
                                    isDirty = 0
                        else:
                            isDirty = 1
                    else:
                        if self.NumberOfSections <= 2:
                            if self.ImageVersion <= 3420:
                                isDirty = 1
                            else:
                                isDirty = 0
                        else:
                            isDirty = 1
            else:
                if self.ImageVersion <= 0:
                    if self.ExportSize <= 23330:
                        if self.IatRVA <= 98304:
                            if self.NumberOfSections <= 3:
                                isDirty = 1
                            else:
                                if self.IatRVA <= 53872:
                                    if self.VirtualSize2 <= 17.5:
                                        isDirty = 1
                                    else:
                                        if self.NumberOfSections <= 10.5:
                                            if self.ResourceSize <= 3103192:
                                                if self.ExportSize <= 10858.5:
                                                    if self.VirtualSize2 <= 116016.5:
                                                        isDirty = 0
                                                    else:
                                                        isDirty = 1
                                                else:
                                                    isDirty = 0
                                            else:
                                                isDirty = 1
                                        else:
                                            isDirty = 1
                                else:
                                    if self.ExportSize <= 273:
                                        isDirty = 1
                                    else:
                                        if self.ResourceSize <= 1016:
                                            isDirty = 1
                                        else:
                                            isDirty = 0
                        else:
                            isDirty = 0
                    else:
                        isDirty = 1
                else:
                    if self.ExportSize <= 1006718985:
                        isDirty = 0
                    else:
                        isDirty = 1
        else:
            if self.ResourceSize <= 545:
                if self.ExportSize <= 92:
                    if self.NumberOfSections <= 4:
                        isDirty = 0
                    else:
                        if self.ImageVersion <= 6005:
                            if self.ExportSize <= 6714:
                                isDirty = 1
                            else:
                                isDirty = 0
                        else:
                            isDirty = 0
            else:
                if self.IatRVA <= 94208:
                    if self.NumberOfSections <= 5:
                        if self.ExportSize <= 0:
                            if self.NumberOfSections <= 4:
                                if self.IatRVA <= 13504:
                                    if self.ImageVersion <= 353:
                                        if self.NumberOfSections <= 3:
                                            if self.IatRVA <= 6144:
                                                if self.IatRVA <= 2048:
                                                    if self.ResourceSize <= 934:
                                                        isDirty = 1
                                                    else:
                                                        if self.VirtualSize2 <= 2728:
                                                            isDirty = 0
                                                        else:
                                                            isDirty = 1
                                                else:
                                                    if self.VirtualSize2 <= 496:
                                                        isDirty = 1
                                                    else:
                                                        isDirty = 0
                                            else:
                                                isDirty = 0
                                        else:
                                            if self.DebugSize <= 41:  # debug here
                                                if self.ResourceSize <= 22720:
                                                    if self.IatRVA <= 2048:
                                                        isDirty = 1
                                                    else:
                                                        if self.VirtualSize2 <= 46:
                                                            isDirty = 0
                                                        else:
                                                            isDirty = 1
                                                else:
                                                    if self.VirtualSize2 <= 43030:
                                                        if self.ResourceSize <= 3898348:
                                                            if self.IatRVA <= 2048:
                                                                isDirty = 1
                                                            else:
                                                                isDirty = 0
                                                        else:
                                                            isDirty = 1
                                                    else:
                                                        isDirty = 0
                                            else:
                                                isDirty = 0
                                    else:
                                        isDirty = 0
                                else:
                                    if self.ResourceSize <= 35328:
                                        if self.ImageVersion <= 4005:
                                            if self.NumberOfSections <= 1.5:
                                                isDirty = 1
                                            else:
                                                isDirty = 0
                                        else:
                                            isDirty = 0
                                    else:
                                        if self.ImageVersion <= 5510:
                                            if self.DebugSize <= 42:
                                                if self.VirtualSize2 <= 144328:
                                                    if self.NumberOfSections <= 3.5:
                                                        isDirty = 0
                                                    else:
                                                        isDirty = 1
                                                else:
                                                    isDirty = 0
                                            else:
                                                isDirty = 0
                                        else:
                                            isDirty = 0
                            else:
                                if self.IatRVA <= 2048:
                                    isDirty = 1
                                else:
                                    isDirty = 0
                        else:
                            isDirty = 0
                    else:
                        if self.IatRVA <= 1054:
                            if self.ExportSize <= 218:
                                if self.IatRVA <= 704:
                                    isDirty = 1
                                else:
                                    if self.NumberOfSections <= 6:
                                        isDirty = 1
                                    else:
                                        isDirty = 0
                            else:
                                if self.ExportSize <= 1006699445:
                                    if self.ImageVersion <= 5510:
                                        if self.ImageVersion <= 500:
                                            isDirty = 1
                                        else:
                                            isDirty = 0
                                    else:
                                        isDirty = 0
                                else:
                                    isDirty = 1
                        else:
                            isDirty = 0
                else:
                    if self.ExportSize <= 0:
                        if self.VirtualSize2 <= 78800:
                            if self.NumberOfSections <= 4:
                                isDirty = 0
                            else:
                                if self.ImageVersion <= 2340:
                                    if self.ResourceSize <= 7328:
                                        isDirty = 1
                                    else:
                                        if self.VirtualSize2 <= 8288.5:
                                            isDirty = 1
                                        else:
                                            if self.NumberOfSections <= 6.5:
                                                isDirty = 0
                                            else:
                                                isDirty = 1
                                else:
                                    isDirty = 0
                        else:
                            if self.ImageVersion <= 5515:
                                isDirty = 1
                            else:
                                isDirty = 0
                    else:
                        if self.IatRVA <= 106496:
                            if self.ResourceSize <= 2800:
                                isDirty = 0
                            else:
                                if self.ImageVersion <= 500:
                                    if self.ResourceSize <= 5360:
                                        if self.NumberOfSections <= 4.5:
                                            isDirty = 0
                                        else:
                                            if self.VirtualSize2 <= 22564.5:
                                                if self.ExportSize <= 191.5:
                                                    if self.DebugSize <= 42:
                                                        if self.ExportSize <= 162.5:
                                                            isDirty = 0
                                                        else:
                                                            if self.VirtualSize2 <= 10682:
                                                                isDirty = 0
                                                            else:
                                                                if self.ResourceSize <= 3412:
                                                                    isDirty = 0
                                                                else:
                                                                    isDirty = 1
                                                    else:
                                                        isDirty = 0
                                                else:
                                                    isDirty = 0
                                            else:
                                                isDirty = 0
                                    else:
                                        isDirty = 0
                                else:
                                    isDirty = 0
                        else:
                            isDirty = 0
        return isDirty

    def runPART(self):
        isDirty = 0
        if self.DebugSize > 0 and self.ResourceSize > 545 and self.IatRVA <= 94208 and self.NumberOfSections <= 5 and self.ExportSize > 0 and self.NumberOfSections > 3:
            isDirty = 0
        elif self.DebugSize <= 0 and self.ImageVersion <= 4900 and self.ExportSize <= 71 and self.ImageVersion <= 520 and self.VirtualSize2 > 130 and self.IatRVA <= 24576:
            isDirty = 1
        elif self.DebugSize <= 0 and self.ImageVersion <= 4900 and self.ExportSize <= 211 and self.ResourceSize <= 32272 and self.NumberOfSections <= 10 and self.VirtualSize2 <= 5 and self.ImageVersion <= 3420:
            isDirty = 1
        elif self.DebugSize > 0 and self.ResourceSize > 598 and self.VirtualSize2 <= 105028 and self.VirtualSize2 > 1 and self.ImageVersion > 5000:
            isDirty = 0
        elif self.IatRVA <= 0 and self.ImageVersion > 4180 and self.ResourceSize > 2484:
            isDirty = 0
        elif self.DebugSize <= 0 and self.NumberOfSections <= 1 and self.ResourceSize > 501:
            isDirty = 0
        elif self.DebugSize <= 0 and self.ExportSize <= 211 and self.NumberOfSections > 2 and self.ImageVersion > 1000 and self.ResourceSize <= 12996:
            isDirty = 1
        elif self.DebugSize <= 0 and self.ExportSize <= 211 and self.NumberOfSections > 2 and self.ResourceSize > 0 and self.VirtualSize2 > 1016:
            isDirty = 1
        elif self.NumberOfSections > 8 and self.VirtualSize2 <= 2221:
            isDirty = 1
        elif self.ResourceSize <= 736 and self.NumberOfSections <= 3:
            isDirty = 1
        elif self.NumberOfSections <= 3 and self.IatRVA > 4156:
            isDirty = 0
        elif self.ImageVersion <= 6000 and self.ResourceSize <= 523 and self.IatRVA > 0 and self.ExportSize <= 95:
            isDirty = 1
        elif self.ExportSize <= 256176 and self.DebugSize > 0 and self.ImageVersion <= 5450 and self.IatRVA > 1664 and self.ResourceSize <= 2040 and self.DebugSize <= 41:
            isDirty = 0
        elif self.ExportSize <= 256176 and self.ImageVersion > 5450:
            isDirty = 0
        elif self.ExportSize > 256176:
            isDirty = 1
        elif self.ImageVersion > 0 and self.ResourceSize > 298216 and self.IatRVA <= 2048:
            isDirty = 1
        elif self.ImageVersion > 0 and self.ExportSize > 74 and self.DebugSize > 0:
            isDirty = 0
        elif self.ImageVersion > 0 and self.VirtualSize2 > 4185 and self.ResourceSize <= 215376 and self.IatRVA <= 2048 and self.NumberOfSections <= 5:
            isDirty = 0
        elif self.ImageVersion > 1010 and self.DebugSize <= 56 and self.VirtualSize2 <= 215376:
            isDirty = 0
        elif self.ExportSize > 258 and self.NumberOfSections > 3 and self.DebugSize > 0:
            isDirty = 0
        elif self.ExportSize > 262 and self.ImageVersion > 0 and self.NumberOfSections > 7:
            isDirty = 0
        elif self.DebugSize > 41 and self.NumberOfSections <= 4:
            isDirty = 0
        elif self.ExportSize <= 262 and self.NumberOfSections > 3 and self.VirtualSize2 <= 37:
            isDirty = 1
        elif self.VirtualSize2 > 40 and self.ExportSize <= 262 and self.DebugSize <= 0 and self.ImageVersion <= 353 and self.ExportSize <= 142:
            isDirty = 1
        elif self.VirtualSize2 > 72384 and self.VirtualSize2 <= 263848:
            isDirty = 1
        elif self.IatRVA > 106496 and self.IatRVA <= 937984 and self.DebugSize > 0 and self.ResourceSize > 4358:
            isDirty = 0
        elif self.VirtualSize2 <= 64 and self.IatRVA <= 2048 and self.DebugSize <= 0 and self.ImageVersion <= 353 and self.ExportSize <= 0 and self.VirtualSize2 <= 4 and self.NumberOfSections <= 2:
            isDirty = 0
        elif self.DebugSize <= 0 and self.NumberOfSections <= 4 and self.IatRVA > 45548:
            isDirty = 1
        elif self.DebugSize > 0 and self.DebugSize <= 56 and self.IatRVA <= 94208 and self.ResourceSize <= 4096:
            isDirty = 1
        elif self.DebugSize <= 0 and self.IatRVA <= 98304 and self.NumberOfSections > 6 and self.ResourceSize <= 864 and self.ExportSize > 74 and self.ImageVersion > 353 and self.ExportSize <= 279:
            isDirty = 0
        elif self.DebugSize <= 0 and self.IatRVA <= 98304 and self.NumberOfSections <= 2 and self.ResourceSize <= 1264128:
            isDirty = 1
        elif self.VirtualSize2 <= 64 and self.IatRVA <= 2048 and self.DebugSize > 0:
            isDirty = 0
        elif self.ExportSize <= 276 and self.NumberOfSections > 5 and self.ResourceSize <= 1076:
            isDirty = 0
        elif self.DebugSize > 0 and self.IatRVA <= 94208 and self.ExportSize <= 82 and self.DebugSize <= 56 and self.NumberOfSections > 2 and self.ImageVersion <= 2340 and self.ResourceSize <= 118280 and self.VirtualSize2 > 5340:
            isDirty = 0
        elif self.DebugSize > 0 and self.ImageVersion <= 2340 and self.DebugSize <= 56 and self.NumberOfSections > 3 and self.VirtualSize2 > 360 and self.NumberOfSections <= 5:
            isDirty = 1
        elif self.IatRVA > 37380 and self.ImageVersion <= 0 and self.NumberOfSections <= 5 and self.VirtualSize2 > 15864:
            isDirty = 0
        elif self.DebugSize <= 0 and self.VirtualSize2 <= 80 and self.IatRVA <= 4096 and self.ExportSize <= 0 and self.VirtualSize2 > 4 and self.VirtualSize2 <= 21:
            isDirty = 0
        elif self.DebugSize <= 0:
            isDirty = 1
        elif self.ExportSize <= 82 and self.DebugSize <= 56 and self.NumberOfSections <= 5 and self.NumberOfSections > 2 and self.IatRVA <= 6144 and self.ImageVersion > 2340:
            isDirty = 0
        elif self.ImageVersion > 2340:
            isDirty = 1
        elif self.ResourceSize > 5528:
            isDirty = 0
        else:
            isDirty = 1
        return isDirty

    def runRidor(self):
        isDirty = 0
        if self.DebugSize <= 14 and self.ImageVersion <= 760 and self.VirtualSize2 > 992 and self.ExportSize <= 80.5:
            isDirty = 1
        elif self.DebugSize <= 14 and self.ImageVersion <= 4525 and self.ExportSize <= 198.5 and self.ResourceSize <= 7348 and self.VirtualSize2 <= 6 and self.ResourceSize > 1773:
            isDirty = 1
        elif self.DebugSize <= 14 and self.ImageVersion <= 4950 and self.ExportSize <= 56 and self.IatRVA > 256 and self.VirtualSize2 > 42 and self.NumberOfSections > 3.5:
            isDirty = 1
        elif self.DebugSize <= 14 and self.ImageVersion <= 4950 and self.VirtualSize2 <= 6 and self.ResourceSize > 17302:
            isDirty = 1
        elif self.DebugSize <= 14 and self.NumberOfSections >= 2.5 and self.ResourceSize <= 1776 and self.IatRVA <= 6144 and self.ExportSize <= 219.5 and self.VirtualSize2 > 2410 and self.VirtualSize2 <= 61224:
            isDirty = 1
        elif self.DebugSize <= 14 and self.NumberOfSections >= 2.5 and self.ExportSize <= 198 and self.ResourceSize > 8 and self.VirtualSize2 > 83 and self.ResourceSize <= 976:
            isDirty = 1
        elif self.DebugSize <= 14 and self.NumberOfSections >= 2.5 and self.ResourceSize > 1418 and self.IatRVA > 6144 and self.VirtualSize2 <= 4:
            isDirty = 1
        elif self.DebugSize <= 14 and self.VirtualSize2 > 14 and self.NumberOfSections > 4.5 and self.ResourceSize > 1550 and self.VirtualSize2 <= 2398:
            isDirty = 1
        elif self.DebugSize <= 14 and self.VirtualSize2 > 14 and self.NumberOfSections > 4.5 and self.ExportSize > 138.5 and self.ImageVersion > 1005:
            isDirty = 1
        elif self.ImageVersion <= 5005 and self.DebugSize <= 14 and self.VirtualSize2 > 14 and self.NumberOfSections <= 4.5:
            isDirty = 1
        elif self.ImageVersion <= 5005 and self.DebugSize <= 14 and self.ImageVersion <= 5 and self.NumberOfSections > 3.5 and self.ExportSize <= 164.5 and self.IatRVA <= 73728 and self.ResourceSize <= 8722:
            isDirty = 1
        elif self.ImageVersion <= 5005 and self.DebugSize <= 14 and self.ResourceSize > 21108 and self.ResourceSize <= 37272 and self.ImageVersion <= 760:
            isDirty = 1
        elif self.NumberOfSections > 4.5 and self.ExportSize <= 25.5 and self.ImageVersion > 1505 and self.ResourceSize <= 1020:
            isDirty = 1
        elif self.ImageVersion <= 1500 and self.NumberOfSections > 5.5 and self.ExportSize <= 101 and self.ResourceSize <= 3168:
            isDirty = 1
        elif self.ImageVersion <= 3025 and self.DebugSize <= 14 and self.ResourceSize > 1182 and self.VirtualSize2 > 164 and self.ExportSize <= 330.5:
            isDirty = 1
        elif self.ImageVersion <= 1010 and self.ResourceSize > 2352 and self.VirtualSize2 > 115254 and self.VirtualSize2 <= 153258:
            isDirty = 1
        elif self.ImageVersion <= 1500 and self.NumberOfSections > 5.5 and self.ImageVersion <= 500 and self.ExportSize <= 164 and self.IatRVA <= 2048:
            isDirty = 1
        elif self.ImageVersion <= 1010 and self.ResourceSize <= 474 and self.IatRVA > 26624 and self.VirtualSize2 > 1802 and self.IatRVA <= 221348:
            isDirty = 1
        elif self.ImageVersion <= 2500 and self.DebugSize <= 14 and self.ResourceSize > 78678 and self.ResourceSize <= 120928 and self.NumberOfSections <= 4:
            isDirty = 1
        elif self.ImageVersion <= 5005 and self.ExportSize <= 25.5 and self.NumberOfSections > 3.5 and self.ResourceSize > 35814 and self.VirtualSize2 > 215352:
            isDirty = 1
        elif self.ImageVersion <= 500 and self.IatRVA <= 2560 and self.NumberOfSections > 3.5 and self.ResourceSize > 648 and self.ResourceSize <= 62291:
            isDirty = 1
        elif self.ExportSize <= 25.5 and self.NumberOfSections > 4.5 and self.VirtualSize2 > 50765 and self.ResourceSize <= 741012 and self.ResourceSize > 2512:
            isDirty = 1
        elif self.ImageVersion <= 1010 and self.ExportSize <= 25.5 and self.VirtualSize2 <= 3278 and self.VirtualSize2 > 1200 and self.ResourceSize > 2032:
            isDirty = 1
        elif self.ResourceSize <= 474 and self.ExportSize <= 76 and self.VirtualSize2 <= 1556 and self.IatRVA <= 2368:
            isDirty = 1
        elif self.ImageVersion <= 1500 and self.VirtualSize2 <= 6 and self.IatRVA > 2048:
            isDirty = 1
        else:
            isDirty = 0
        return isDirty

    def predict(self):
        # Unexpectedly formed files will be classified malicious
        if not self.init_success:
            return 1.0

        if "DebugSize" not in self.__dict__:
            raise Exception("Need to initialize before predicting")

        self.J48 = self.runJ48()
        self.J48Graft = self.runJ48Graft()
        self.PART = self.runPART()
        self.Ridor = self.runRidor()
        return sum([self.J48, self.J48Graft, self.PART, self.Ridor]) / 4.0


class AdobeModel:

    def __init__(self):
        return

    def predict_raw_features(self, raw_features):
        y_pred = []
        for raw_feature_dict in raw_features:
            y_pred.append(AdobeEval(raw_features=raw_feature_dict).predict())
        return y_pred

    def predict_paths(self, paths):
        y_pred = []
        for path in paths:
            y_pred.append(AdobeEval(path=path).predict())
        return y_pred


def vectorize(irow, raw_features_string, X_path, y_path, nrows):
    """
    Vectorize a single sample of raw features and write to a large numpy file
    """
    raw_features = json.loads(raw_features_string)
    feature_vector = AdobeEval(raw_features=raw_features).feature_vector()

    y = np.memmap(y_path, dtype=np.float32, mode="r+", shape=nrows)
    y[irow] = raw_features["label"]

    X = np.memmap(X_path, dtype=np.float32, mode="r+", shape=(nrows, AdobeEval.dim))
    X[irow] = feature_vector


def vectorize_unpack(args):
    """
    Pass through function for unpacking vectorize arguments
    """
    return vectorize(*args)


def vectorize_subset(X_path, y_path, raw_feature_paths, nrows):
    """
    Vectorize a subset of data and write it to disk
    """
    # Create space on disk to write features to
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, AdobeEval.dim))
    y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=nrows)
    del X, y

    # Distribute the vectorization work
    pool = multiprocessing.Pool()
    argument_iterator = ((irow, raw_features_string, X_path, y_path, nrows)
                         for irow, raw_features_string in enumerate(ember.raw_feature_iterator(raw_feature_paths)))
    for _ in tqdm.tqdm(pool.imap_unordered(vectorize_unpack, argument_iterator), total=nrows):
        pass


def create_vectorized_features(data_dir):
    """
    Create feature vectors from raw features and write them to disk
    """
    print("Vectorizing training set")
    X_path = os.path.join(data_dir, "X_train_adobe.dat")
    y_path = os.path.join(data_dir, "y_train_adobe.dat")
    raw_feature_paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(6)]
    nrows = sum([1 for fp in raw_feature_paths for line in open(fp)])
    vectorize_subset(X_path, y_path, raw_feature_paths, nrows)

    print("Vectorizing test set")
    X_path = os.path.join(data_dir, "X_test_adobe.dat")
    y_path = os.path.join(data_dir, "y_test_adobe.dat")
    raw_feature_paths = [os.path.join(data_dir, "test_features.jsonl")]
    nrows = sum([1 for fp in raw_feature_paths for line in open(fp)])
    vectorize_subset(X_path, y_path, raw_feature_paths, nrows)


def read_vectorized_features(data_dir, subset=None):
    """
    Read vectorized features into memory mapped numpy arrays
    """
    if subset is not None and subset not in ["train", "test"]:
        return None

    ndim = AdobeEval.dim
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    if subset is None or subset == "train":
        X_train_path = os.path.join(data_dir, "X_train_adobe.dat")
        y_train_path = os.path.join(data_dir, "y_train_adobe.dat")
        y_train = np.memmap(y_train_path, dtype=np.float32, mode="r")
        N = y_train.shape[0]
        X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(N, ndim))
        if subset == "train":
            return X_train, y_train

    if subset is None or subset == "test":
        X_test_path = os.path.join(data_dir, "X_test_adobe.dat")
        y_test_path = os.path.join(data_dir, "y_test_adobe.dat")
        y_test = np.memmap(y_test_path, dtype=np.float32, mode="r")
        N = y_test.shape[0]
        X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(N, ndim))
        if subset == "test":
            return X_test, y_test

    return X_train, y_train, X_test, y_test


def optimize_model(data_dir):
    """
    Run a grid search to find the best LightGBM parameters
    """
    # Read data
    X_train, y_train = read_vectorized_features(data_dir, subset="train")

    # Filter unlabeled data
    train_rows = (y_train != -1)

    # read training dataset
    X_train = X_train[train_rows]
    y_train = y_train[train_rows]

    # score by roc auc
    # we're interested in low FPR rates, so we'll consider only the AUC for FPRs in [0,5e-3]
    score = make_scorer(roc_auc_score, max_fpr=5e-3)

    # define search grid
    param_grid = {
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
        'num_iterations': [500, 1000],
        'learning_rate': [0.005, 0.05],
        'num_leaves': [512, 1024, 2048],
        'feature_fraction': [0.5, 0.8, 1.0],
        'bagging_fraction': [0.5, 0.8, 1.0]
    }
    model = lgb.LGBMClassifier(boosting_type="gbdt", n_jobs=-1, silent=True)

    # each row in X_train appears in chronological order of "appeared"
    # so this works for progrssive time series splitting
    progressive_cv = TimeSeriesSplit(n_splits=3).split(X_train)

    grid = GridSearchCV(estimator=model, cv=progressive_cv, param_grid=param_grid, scoring=score, n_jobs=1, verbose=3)
    grid.fit(X_train, y_train)

    print(grid.best_params_)
    json.dump(grid.best_params_, open("adobe_best_params.json", "w"))


def train_model(data_dir):
    """
    Train the LightGBM model from the EMBER dataset from the vectorized features
    """
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_iterations': 500,
        'learning_rate': 0.005,
        'num_leaves': 2048,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
    }

    # Read data
    X_train, y_train = read_vectorized_features(data_dir, "train")

    # Filter unlabeled data
    train_rows = (y_train != -1)

    # Train
    lgbm_dataset = lgb.Dataset(X_train[train_rows], y_train[train_rows])
    lgbm_model = lgb.train(params, lgbm_dataset)
    lgbm_model.save_model(os.path.join(data_dir, "adobe_model_optimized.txt"))

    return lgbm_model


def find_badly_classified_families(data_dir):
    """
    Find classes that are classified poorly by the benchmark model
    """

    emberdf = ember.read_metadata(data_dir)
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(data_dir)
    lgbm_model = lgb.Booster(model_file=os.path.join(data_dir, "ember_model_2018.txt"))
    y_test_pred = lgbm_model.predict(X_test)
    y_train_pred = lgbm_model.predict(X_train)
    emberdf["y_pred_ember"] = np.hstack((y_train_pred, y_test_pred))

    nclasses = 1000
    detection_rates = []
    avclass_counts = emberdf[emberdf.subset == "train"].avclass.value_counts()
    for i, family in enumerate(avclass_counts.index[:nclasses]):
        familydf = emberdf[(emberdf.subset == "test") & (emberdf.avclass == family)]
        detection_rates.append((familydf.y_pred_ember > 0.8336).sum() / len(familydf))

    famdf = pd.DataFrame({"avclass": avclass_counts.index[:nclasses], "detection_rates": detection_rates})
    badly_classified_families = list(famdf[famdf.detection_rates < 0.96498].avclass)
    open(data_dir + "/badly_classified_families.txt", "w").write("\n".join(badly_classified_families))


def train_weighted_model(data_dir):
    """
    Train the LightGBM model from the EMBER dataset from the vectorized features
    """
    params = {
        "boosting": "gbdt",
        "objective": "binary",
        "num_iterations": 1000,
        "learning_rate": 0.05,
        "num_leaves": 2048,
        "max_depth": 15,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.5
    }

    # Read data
    badly_classified_families = open(data_dir + "/badly_classified_families.txt").read().strip().split("\n")
    emberdf = ember.read_metadata(data_dir)
    X_train, y_train = ember.read_vectorized_features(data_dir, "train")
    w_train = np.array(emberdf[emberdf.subset == "train"].avclass.isin(badly_classified_families)) + 1

    # Filter unlabeled data
    train_rows = (y_train != -1)

    # Train
    lgbm_dataset = lgb.Dataset(X_train[train_rows], y_train[train_rows], weight=w_train[train_rows])
    lgbm_model = lgb.train(params, lgbm_dataset)
    lgbm_model.save_model(os.path.join(data_dir, "ember_model_2018_weighted.txt"))

    return lgbm_model


def train_multiple(data_dir):
    """
    Train a bunch of models to explore how different they are
    """
    params = {
        "boosting": "gbdt",
        "objective": "binary",
        "num_iterations": 1000,
        "learning_rate": 0.05,
        "num_leaves": 2048,
        "feature_fraction": 0.5,
        "bagging_fraction": 1.0,
        "max_depth": 15,
        "min_data_in_leaf": 50
    }
    for i in range(10):
        lgbm_model = ember.train_model(data_dir, params, 2)
        lgbm_model.save_model(os.path.join(data_dir, f"ember_model_2018_random{i}.txt"))


def find_disagreements(data_dir, samples_dir):
    """
    A bunch of samples will have different Adobe features from EMBER than from the original implementation. This is
    due to pefile and lief parsing sections in differnet orders. Sometimes, the Virtual Size of the second section
    will differ because pefile and lief disagree about which is the second section.
    """
    for jsonl_file in glob.glob(f"{data_dir}/*jsonl"):
        for line in open(jsonl_file):
            raw_feature_dict = json.loads(line)
            sha256 = raw_feature_dict["sha256"]
            path = f"{samples_dir}/{sha256[0]}/{sha256[1]}/{sha256[2]}/{sha256}"
            eval_rf = AdobeEval(raw_features=raw_feature_dict)
            eval_p = AdobeEval(path=path)
            if eval_rf.init_success and eval_p.init_success and eval_p != eval_rf:
                print(sha256)
                for f in AdobeEval.ordered_features:
                    if eval_p.__dict__[f] != eval_rf.__dict__[f]:
                        print(f"{f+':':<18}pefile: {eval_p.__dict__[f]} lief: {eval_rf.__dict__[f]}")
