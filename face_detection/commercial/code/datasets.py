import glob
import json
from pathlib import Path
import pandas as pd
import numpy as np

import os
import numpy as np
import pandas as pd
from itertools import groupby

CAP = 1500
SEED = 1234
np.random.seed(SEED)

class dataset:
    def write_unique_ids(self, out_file):
        """
        Write the unique IDs to a file, but add a self.prefix to each element of the array.
        For example, if self.unique_ids is
            ['image_1.jpg', 'image_2.jpg']
        then if the self.prfix is './folder/', then out_file would be written as
            ./folder/image_1.jpg
            ./folder/image_2.jpg
        """
        with open(out_file,'w') as f:
            f.writelines([self.prefix+x+'\n' for x in self.unique_ids])
        return

    def read_unique_ids(self, in_file, prefix=None):
        """
        Read the unique IDs from in_file, but remove a self.prefix from each element of the array.
        For example, if the in_file is
            ./folder/image_1.jpg
            ./folder/image_2.jpg
        and the self.prefix is './folder/', then self.unique_ids would be written as
            ['image_1.jpg', 'image_2.jpg']
        """
        if prefix is None:
            prefix = self.prefix
        with open(in_file) as f:
            self.unique_ids = [x.strip().replace(prefix, '') for x in f]
        return


class adience_dataset(dataset):
    def __init__(self, metadata_folder='./'):
        """
        Create the Adience Dataset class.
        Ususally run as:
            adi = Adience_dataset(metadata_folder)
            adi.select_unique_ids()
            adi.write_unique_ids('adience_images.txt')
        Or if the unique_ids have already been created:
            adi = Adience_dataset(metadata_folder)
            adi.read_unique_ids('adience_images.txt')


        """
        self.metadata = self.load_metadata(metadata_folder)
        self.prefix = 'data/adience/faces/'
        return

    def load_metadata(self, metadata_folder):
        def adience_resolve_class_label(age):
            """
            Given an age, what is the age group?
            """
            if age == '(0, 2)' or age == '2':
                age_id = 0
            elif age == '(4, 6)' or age == '3':
                age_id = 1
            elif age == '(8, 12)' or age == '(8, 23)' or age == '13':
                age_id = 2
            elif age == '(15, 20)' or age == '22':
                age_id = 3
            elif age == '(25, 32)' or age == '(27, 32)' or age in ['23', '29', '34', '35']:
                age_id = 4
            elif age == '(38, 42)' or age == '(38, 43)' or age == '(38, 48)' or age in ['36', '42', '45']:
                age_id = 5
            elif age == '(48, 53)' or age in ['46', '55']:
                age_id = 6
            elif age == '(60, 100)' or age in ['57', '58']:
                age_id = 7
            else:
                raise ValueError("Not sure how to handle this age: {}".format(age))

            return age_id

        if metadata_folder[-1] == '/':
            metadata_folder = metadata_folder[:-1]

        fold_0 = pd.read_csv(f'{metadata_folder}/fold_0_data.txt', sep='\t')
        fold_1 = pd.read_csv(f'{metadata_folder}/fold_1_data.txt', sep='\t')
        fold_2 = pd.read_csv(f'{metadata_folder}/fold_2_data.txt', sep='\t')
        fold_3 = pd.read_csv(f'{metadata_folder}/fold_3_data.txt', sep='\t')
        fold_4 = pd.read_csv(f'{metadata_folder}/fold_4_data.txt', sep='\t')

        # get only those data that have an age and gender is m or f
        fold_0 = fold_0[np.logical_and(fold_0['age'] != 'None',
                           np.logical_or(fold_0['gender'] == 'm', fold_0['gender'] == 'f'))]
        fold_1 = fold_1[np.logical_and(fold_1['age'] != 'None',
                           np.logical_or(fold_1['gender'] == 'm', fold_1['gender'] == 'f'))]
        fold_2 = fold_2[np.logical_and(fold_2['age'] != 'None',
                           np.logical_or(fold_2['gender'] == 'm', fold_2['gender'] == 'f'))]
        fold_3 = fold_3[np.logical_and(fold_3['age'] != 'None',
                           np.logical_or(fold_3['gender'] == 'm', fold_3['gender'] == 'f'))]
        fold_4 = fold_4[np.logical_and(fold_4['age'] != 'None',
                           np.logical_or(fold_4['gender'] == 'm', fold_4['gender'] == 'f'))]

        adience = pd.concat([fold_0,fold_1,fold_2,fold_3,fold_4])
        adience['age_group'] = adience.age.apply(adience_resolve_class_label)
        adience['ImageID'] = 'coarse_tilt_aligned_face.'+adience['face_id'].astype(str) +'.'+ adience['original_image'].apply(lambda x: x.replace('.jpg',''))
        return adience


    def select_unique_ids(self):
        """
        Randomly select images from the Adience dataset to be included in the
        experiments. Make sure that there are at least #CAP number of images
        in each intersection for age and gender groups.
        """
        adience = self.metadata
        adi_ids = []
        for gg in set(adience['gender']):
                for ag in set(adience['age_group']):
                    try:
                        idx = np.logical_and(adience['gender'] == gg,adience['age_group'] == ag)
                        intersection_ids = list(adience[idx]['user_id'] +
                                                '/coarse_tilt_aligned_face.' +
                                                adience[idx]['face_id'].astype(str) +
                                                '.' + adience[idx]['original_image'])
                        if len(intersection_ids) <= CAP:
                            adi_ids += intersection_ids
                        else:
                            x = list(np.random.choice(intersection_ids, CAP, replace=False))
                            adi_ids += x

                    except:
                        continue
        self.unique_ids = adi_ids
        return adi_ids

class ccd_dataset(dataset):
    def __init__(self, metadata_folder='./'):
        """
        Create the CCD Dataset class.
        Ususally run as:
            c = CCD_dataset(metadata_folder)
            c.select_unique_ids()
            c.write_unique_ids('ccd_images.txt')
        Or if the unique_ids have already been created:
            c = CCD_dataset(metadata_folder)
            c.read_unique_ids('ccd_images.txt')


        """
        self.metadata = self.load_metadata(metadata_folder)
        self.prefix = 'data/CCD/frames/'
        return

    def load_metadata(self, metadata_folder):
        if metadata_folder[-1] == '/':
            metadata_folder = metadata_folder[:-1]

        with open(metadata_folder+'/CasualConversations.json') as f:
            ccd = json.load(f)

        rows = []
        for k in ccd.keys():
            i = int(k)
            dark = ccd[k]['dark_files']
            all_ = ccd[k]['files']
            age = ccd[k]['label']['age']
            gender = ccd[k]['label']['gender']
            skin = int(ccd[k]['label']['skin-type'])<3
            age = int(age) if age != 'N/A' else -1
            if age in range(18):
                age_id = 0
            elif age in range(18,45):
                age_id = 1
            elif age in range(45,65):
                age_id = 2
            elif age in range(65,122):
                age_id = 3
            for f in all_:
                png = f.replace('.MP4','.png')
                if f in dark:
                    rows += [[png,1,age_id,age,gender,skin]]
                else:
                    rows += [[png,0,age_id,age, gender,skin]]

        ccd = pd.DataFrame(rows, columns=['ImageID','isDark', 'Age', 'Age_Numeric','Gender', 'Skin'])
        return ccd


    def select_unique_ids(self):
        """
        Randomly select images from the CCD dataset to be included in the
        experiments. Make sure that there are at least #CAP number of images
        in each intersection for age, gender, lighting condition, and skin groups.
        """
        ccd = self.metadata
        ccd_ids = []
        for dg in set(ccd['isDark']):
            for gg in set(ccd['Gender']):
                for sg in set(ccd['Skin']):
                    for ag in set(ccd['Age']):
                        try:
                            intersection_ids = list(ccd[np.logical_and(ccd['isDark'] == dg,
                                                        np.logical_and(ccd['Gender'] == gg,
                                                        np.logical_and(ccd['Skin']   == sg,
                                                                       ccd['Age'] == ag)))]['ImageID'])
                            if len(intersection_ids) <= CAP:
                                ccd_ids += intersection_ids
                            else:
                                x = list(np.random.choice(intersection_ids, CAP, replace=False))
                                ccd_ids += x

                        except:
                            continue
        self.unique_ids = ccd_ids
        return ccd_ids

class miap_dataset(dataset):
    def __init__(self, metadata_folder='./'):
        """
        Create the MAIP Dataset class.
        Ususally run as:
            miap = MIAP_dataset(metadata_folder)
            miap.select_unique_ids()
            miap.write_unique_ids('miap_images.txt')
        Or if the unique_ids have already been created:
            miap = MIAP_dataset(metadata_folder)
            miap.read_unique_ids('miap_images.txt')


        """
        self.metadata = self.load_metadata(metadata_folder)
        self.prefix = 'data/miap/images/'
        return

    def load_metadata(self, metadata_folder):
        "metadata_folder should not have a trailing /"
        if metadata_folder[-1] == '/':
            metadata_folder = metadata_folder[:-1]
        miap_test = pd.read_csv(f'{metadata_folder}/open_images_extended_miap_boxes_test.csv')
        miap_train = pd.read_csv(f'{metadata_folder}/open_images_extended_miap_boxes_train.csv')
        miap_val = pd.read_csv(f'{metadata_folder}/open_images_extended_miap_boxes_val.csv')

        miap_test['ImageID'] = miap_test['ImageID'].apply(lambda x: 'test/'+x)
        miap_train['ImageID'] = miap_train['ImageID'].apply(lambda x: 'train/'+x)
        miap_val['ImageID'] = miap_val['ImageID'].apply(lambda x: 'validation/'+x)

        miap = pd.concat([miap_test,miap_train,miap_val])
        return miap


    def select_unique_ids(self):
        """
        First only select those MIAP images that have 1 object in them.
        Then randomly select images to be included in the
        experiments. Make sure that there are at least #CAP number of images
        in each intersection for age and gender groups.
        """
        miap = self.metadata
        miap_single = miap[miap.ImageID.isin(list(miap_single[miap_single == 1].index))]
        miap_ids = []
        for gp in set(miap_single['GenderPresentation']):
            for ap in set(miap_single['AgePresentation']):
                try:
                    intersection_ids = list(miap_single[np.logical_and(miap_single['GenderPresentation'] == gp,
                                                                       miap_single['AgePresentation'] == ap)]['ImageID'])
                    if group[gp][ap] <= CAP:
                        miap_ids += intersection_ids
                    else:
                        x = list(np.random.choice(intersection_ids, CAP, replace=False))
                        miap_ids += x

                except:
                    continue
        self.unique_ids = miap_ids
        return miap_ids

class utk_dataset(dataset):
    def __init__(self, utkface_filenames = 'utkface_images.txt'):
        """
        Create the UTK Dataset class.
        Ususally run as:
            utk = UTK_dataset(metadata_folder)
            utk.select_unique_ids()
            utk.write_unique_ids('utk_images.txt')
        Or if the unique_ids have already been created:
            utk = UTK_dataset(metadata_folder)
            utk.read_unique_ids('utk_images.txt')


        """
        self.metadata = self.load_metadata(utkface_filenames)
        self.prefix = ''
        return

    def load_metadata(self, utkface_filenames):
        """
        The metadata for the UTK dataset are in the file names, so pass a list of utk files
        Example:
            data/utkface/UTKface_inthewild/part1/100_1_0_20170110183726390.jpg
            data/utkface/UTKface_inthewild/part1/100_1_2_20170105174847679.jpg
            ...
        """

        def utk_resolve_age_label(file):
            x = file.split('_')
            if len(x) != 4:
                return -1
            age = int(file.split('_')[0])
            if age in range(18):
                age_id = 0
            elif age in range(18,45):
                age_id = 1
            elif age in range(45,65):
                age_id = 2
            elif age in range(65,122):
                age_id = 3
            else:
                raise ValueError("Not sure how to handle this age: {}".format(age))

            return age_id

        def utk_resolve_gender_label(file):
            x = file.split('_')
            return int(x[1]) if len(x)==4 and len(x[1]) else -1

        def utk_resolve_race_label(file):
            x = file.split('_')
            return int(x[2]) if len(x)==4 else -1


        with open(utkface_filenames, 'r') as f:
            files = [x.strip() for x in f]
        utk = pd.DataFrame(files, columns=['filename'])
        utk['ImageID'] = utk['filename'].apply(lambda x: os.path.basename(x))
        utk['age'] = utk['ImageID'].apply(utk_resolve_age_label)
        utk['gender'] = utk['ImageID'].apply(utk_resolve_gender_label)
        utk['race'] = utk['ImageID'].apply(utk_resolve_race_label)
        return utk


    def select_unique_ids(self):
        """
        First only select those MIAP images that have 1 object in them.
        Then randomly select images to be included in the
        experiments. Make sure that there are at least #CAP number of images
        in each intersection for race, age, and gender groups.
        """
        utk = self.metadata
        utk_ids = []
        for gg in set(utk['gender']):
            for rg in set(utk['race']):
                for ag in set(utk['age']):
                    try:
                        intersection_ids = list(utk[np.logical_and(utk['gender'] == gg,
                                                    np.logical_and(utk['race']   == rg,
                                                                   utk['age'] == ag))]['filename'])
                        if len(intersection_ids) <= CAP:
                            utk_ids += intersection_ids
                        else:
                            x = list(np.random.choice(intersection_ids, CAP, replace=False))
                            utk_ids += x

                    except:
                        continue
        self.unique_ids = utk_ids
        return utk_ids
