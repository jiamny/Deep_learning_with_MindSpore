import mindspore as ms

# ----------------------------------
# Example 1.
# ----------------------------------
'''
Import the FileWriter class for file writing.
'''
from mindspore.mindrecord import FileWriter

'''
Define a dataset schema which defines dataset fields and field types.
'''
cv_schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}

'''
Prepare the data sample list to be written based on the user-defined schema format. Binary data of the images is transferred below.
'''
data = [{"file_name": "1.jpg", "label": 0,
         "data": b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff\xd9"},
        {"file_name": "2.jpg", "label": 56,
         "data": b"\xe6\xda\xd1\xae\x07\xb8>\xd4\x00\xf8\x129\x15\xd9\xf2q\xc0\xa2\x91YFUO\x1dsE1\x1ep"},
        {"file_name": "3.jpg", "label": 99,
         "data": b"\xaf\xafU<\xb8|6\xbd}\xc1\x99[\xeaj+\x8f\x84\xd3\xcc\xa0,i\xbb\xb9-\xcdz\xecp{T\xb1\xdb"}]

'''
Adding index fields can accelerate data loading. This step is optional.
'''
indexes = ["file_name", "label"]

'''
Create a FileWriter object, transfer the file name and number of slices, add the schema and index, call the 
write_raw_data API to write data, and call the commit API to generate a local data file.
'''
writer = FileWriter(file_name="../data/test.mindrecord", shard_num=4, overwrite=True)
writer.add_schema(cv_schema_json, "test_schema")
writer.add_index(indexes)
writer.write_raw_data(data)
writer.commit()

'''
For adding data to the existing data format file, call the open_for_append API to open the existing data file, 
call the write_raw_data API to write new data, and then call the commit API to generate a local data file.
'''
writer = FileWriter.open_for_append("../data/test.mindrecord0")
writer.write_raw_data(data)
writer.commit()

# ---------------------------------------------------------------------
# Example 2: Convert a picture in jpg format into a MindRecord dataset
# ---------------------------------------------------------------------
import os
import requests
import tarfile
import zipfile

requests.packages.urllib3.disable_warnings()


def download_dataset(url, target_path):
    """download and decompress dataset"""
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    download_file = url.split("/")[-1]
    if not os.path.exists(download_file):
        res = requests.get(url, stream=True, verify=False)
        if download_file.split(".")[-1] not in ["tgz", "zip", "tar", "gz"]:
            download_file = os.path.join(target_path, download_file)
        with open(download_file, "wb") as f:
            for chunk in res.iter_content(chunk_size=512):
                if chunk:
                    f.write(chunk)
    if download_file.endswith("zip"):
        z = zipfile.ZipFile(download_file, "r")
        z.extractall(path=target_path)
        z.close()
    if download_file.endswith(".tar.gz") or download_file.endswith(".tar") or download_file.endswith(".tgz"):
        t = tarfile.open(download_file)
        names = t.getnames()
        for name in names:
            t.extract(name, target_path)
        t.close()
    print("The {} file is downloaded and saved in the path {} after processing".format(os.path.basename(url),
                                                                                       target_path))


download_dataset("https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/transform.jpg",
                 "../data/images/")
if not os.path.exists("../data/convert_dataset_to_mindrecord/"):
    os.makedirs("../data/convert_dataset_to_mindrecord/")

'''
Execute the following code to convert the downloaded transform.jpg into a MindRecord dataset.
'''
# step 1 import class FileWriter
import os
from mindspore.mindrecord import FileWriter

# clean up old run files before in Linux
data_path = '../data/convert_dataset_to_mindrecord/'
os.system('rm -f {}test.*'.format(data_path))

# import FileWriter class ready to write data
data_record_path = '../data/convert_dataset_to_mindrecord/test.mindrecord'
writer = FileWriter(file_name=data_record_path, shard_num=4, overwrite=True)

# define the data type
data_schema = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
writer.add_schema(data_schema, "test_schema")

# prepeare the data contents
file_name = "../data/images/transform.jpg"
with open(file_name, "rb") as f:
        bytes_data = f.read()
data = [{"file_name": "transform.jpg", "label": 1, "data": bytes_data}]

# add index field
indexes = ["file_name", "label"]
writer.add_index(indexes)

# save data to the files
writer.write_raw_data(data)
writer.commit()

# --------------------------------------------------
# Loading MindRecord Dataset
# --------------------------------------------------
'''
Import the dataset for dataset loading.
'''
import mindspore.dataset as ds

'''
Use the MindDataset to load the MindRecord dataset.
'''
data_set = ds.MindDataset(dataset_files="../data/test.mindrecord0")     # read full dataset
count = 0
for item in data_set.create_dict_iterator(output_numpy=True):
    print("sample: {}".format(item))
    count += 1
print("Got {} samples".format(count))




exit(0)
