import os, sys, optparse, argparse, subprocess, json, ROOT, h5py, torch
import numpy as np
import glob

def GetSize(elem):
  return elem[0].size()[0]

if __name__=='__main__':

  usage = 'usage: %prog [options]'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('--dataset', dest='dataset', help='dataset[1/2/3]',default=1, type=str)
  parser.add_argument("--test", action="store_true")
  parser.add_argument('--store_geometric', dest = 'store_geometric', action = "store_true")
  parser.add_argument("--merge", action="store_true")
  args = parser.parse_args()

  dataset_directory = "/eos/user/t/tihsu/database/ML_hackthon/"
  TaskList = { "dataset1": [
                {"particle": "photon", "xml": "binning_dataset_1_photons.xml", "data": ["dataset_1_photons_1.hdf5", "dataset_1_photons_2.hdf5"]},
                {"particle": "pion", "xml": "binning_dataset_1_pions.xml", "data": ["dataset_1_pions_1.hdf5"]}],
               "dataset2": [
                {"particle": "electron", "xml": "binning_dataset_2.xml", "data": ["dataset_2_1.hdf5","dataset_2_2.hdf5"]}],
               "dataset3": [
                {"particle": "electron", "xml": "binning_dataset_3.xml", "data": ["dataset_3_1.hdf5","dataset_3_2.hdf5","dataset_3_3.hdf5","dataset_3_4.hdf5"]}]
             }

  tag = 'tensor_no_pedding_euclidian'
  file_match = '%s_*.pt'%(args.dataset)
  files_list = glob.glob((os.path.join(dataset_directory, 'tensor', file_match)))
  print((os.path.join(dataset_directory, 'tensor', file_match)))
  os.system('mkdir -p %s'%os.path.join(dataset_directory, 'tmp_ordered'))
  ordered_file_list = []

  max_entries = -1

  for idx, f in enumerate(files_list):
    df, e_inj = torch.load(f)
    e_inj = e_inj.tolist()
    entries    = [x.size()[0] for x in df] 
    if max(entries) > max_entries: max_entries = max(entries)

    ordered_result = sorted(zip(df, e_inj, entries), key=lambda x: x[2])
    ordered_df = [ x for x,y,z in ordered_result]
    ordered_e_inj = [ y for x,y,z in ordered_result]
    ordered_entries = [ z for x,y,z in ordered_result]

    torch.save([ordered_df, torch.tensor(ordered_e_inj), ordered_entries], os.path.join(dataset_directory, 'tmp_ordered', 'dataset_%s_%s_%d.pt'%(args.dataset, tag, idx)))
    ordered_file_list.append(os.path.join(dataset_directory, 'tmp_ordered', 'dataset_%s_%s_%d.pt'%(args.dataset, tag, idx)))
 
  print('finish first stage ordering')


  df_temp = []
  e_inj_temp = []
  entry_temp = []
  current_nentry = 1
  ntotal_file = len(files_list) * 2
  nEntry_segment = int(max_entries / ntotal_file)

  Entry_start = 1
  Entry_end   = nEntry_segment
  while(Entry_end < max_entries + nEntry_segment):
    for idx, f in enumerate(ordered_file_list):
      df, e_inj, entries = torch.load(f)
      e_inj = e_inj.tolist()
      for idx_entries, entry in enumerate(entries):
        if entry >=Entry_start and entry <= Entry_end:
          df_temp.append(df[idx_entries])
          print(e_inj[idx_entries])
          e_inj_temp.append(e_inj[idx_entries])
          entry_temp.append(entry)
    e_inj_temp = torch.tensor(e_inj_temp)
    torch.save([df_temp, e_inj_temp, entry_temp], os.path.join(dataset_directory, 'bucketed_tensor', '%s_%s_nentry%dTo%d.pt'%(args.dataset, tag, Entry_start, Entry_end)))
    Entry_start += nEntry_segment
    Entry_end   += nEntry_segment

    df_temp = []
    e_inj_temp = []
    entry_temp = []

  
     

