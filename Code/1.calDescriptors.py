from multiprocessing import freeze_support
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors


def cal_descriptors_by_mordred(mol_list):
    freeze_support()
    # Instantiate an object——All descriptors calculator
    calculator = Calculator(descriptors)
    mordred_pd = calculator.pandas(mol_list)
    return mordred_pd


def main():
    # 读取数据
    data_all_pd = pd.read_csv(r'data_original.csv')
    data_all = data_all_pd.values
    # 获取所有SMILES
    SMILES_list = data_all[:, 9].reshape(1, -1).tolist()[0]

    # 转换成mol对象
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in SMILES_list]

    # AddHs
    molsAddHs_list = [AllChem.AddHs(mol) for mol in mols_list]

    # 计算描述符并保存成pandas Dataframe

    descriptors_pd = cal_descriptors_by_mordred(molsAddHs_list)

    # 保存文件
    descriptors_pd.to_csv('descriptors_A.csv', index=False)


if __name__ == '__main__':
    main()
