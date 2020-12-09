import numpy as np
import random 
import sympy as sym
from itertools import chain, combinations


rank = np.linalg.matrix_rank

def powerset(iterable):
    '''
    与えられたイテラブル(リスト,タプル,集合など)の冪集合を求めます
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def convertFromCoreNumbersMod(x , m):
    '''
    sympy.core.numbers型の分数xをmod mで整数に変換します
    '''
    x_frac = str(x).split('/')
    if len(x_frac) == 1:
        return np.mod(int(x_frac[0]) , m)
    else:
        numer, denom = map(int, x_frac)
        numer = np.mod(numer, m)
        inverse = None
        for i in range(m):
            if np.mod(i * denom, m) == 1:
                inverse = i
                break
        if inverse is None:
            return None
        return np.mod(numer * inverse, m)


class Person(object):
    '''
    参加者またはディーラーのオブジェクト
    '''
    def __init__(self, name):
        self.name = name
       

class Participant(Person):
    '''
    参加者のオブジェクト
    '''
    def __init__(self, name):
        super().__init__(name)
        self.share = None
        self.label = None


class Dealer(Person):
    '''
    ディーラーのオブジェクト
    '''
    def __init__(self, name, secretsharingscheme):
        super().__init__(name)
        self.SSS = secretsharingscheme
        self.secret = None
        self.t = None
        self.u = None

    def input_secret(self, secret):
        '''
        与えられた秘密を登録します
        '''
        self.secret = secret

    def generate_u(self):
        '''
        秘密鍵uを生成します\n
        u : ug_0 = s を満たすランダムなベクトル\n
        g_0 : 生成行列の一列目\n
        s : 秘密
        '''
        G = self.SSS.G
        m = self.SSS.m
        if self.secret is None:
            s = int(input("秘密値を入力してください :"))
            self.input_secret(s)
        s = self.secret
        v = G.T[0]
        u_labels = [f"u{i}" for i in range(len(v))]
        u_vars = list(map(lambda l: sym.Symbol(l), u_labels)) #symbolを宣言することで計算可能にしている.
        u_labels_valid = []

        equation = -s #方程式
        for v_i, u_i in zip(v, u_vars):
            if v_i != 0:
                u_labels_valid.append(str(u_i))
            equation += v_i * u_i
        solution = sym.solve(equation)
        u_randoms = [(u_v, random.randint(0, m)) for u_v in u_vars]
        
        u_res = []
        for i, u_l in enumerate(u_labels):
            if u_l == u_labels_valid[0]:
                if type(solution[0]) == dict:
                    if type(solution[0][u_vars[i]]) != int:
                        u_res.append(solution[0][u_vars[i]].subs(u_randoms))
                    else:
                        u_res.append(solution[0][u_vars[i]])
                else:
                    u_res.append(solution[0])
            else:
                u_res.append(u_randoms[i][1])
        self.u = np.array(u_res, dtype="int64") % self.SSS.m
      

    def generate_shares(self):
        '''
        シェアtを生成します\n
        t = uG (G : 生成行列, u : 秘密鍵)
        '''
        if self.u is None:
            self.generate_u()
        G = self.SSS.G
        u = self.u
        m = self.SSS.m
        self.t = np.mod(u @ G, m)

    def randomly_distribute_shares(self):
        '''
        参加者にランダムにシェアを配ります
        '''
        P = self.SSS.P
        n = self.SSS.matrix.shape[1]
        share_labels = list(range(1,n))
        share_labels = random.sample(share_labels , n-1)
        for p, i in zip(P, share_labels):
            p.share = self.t[i]
            p.label = i

    def check_decodable(self):
        '''
        参加者のすべての部分集合に対して復号可能かどうかを出力します\n
        表示される結果のバッファを返します
        '''
        buf = []
        P = self.SSS.P
        m = self.SSS.m
        converter = self.SSS.converter
        patterns = list(powerset(P))
        print(patterns)
        for group in patterns:
            print([p.name for p in group])
            buf.append(str([p.name for p in group]))
            v = np.array([p.share for p in group], dtype="int64")
            accessible, coefs = self.SSS.is_decodable(group)
            if accessible:
                coefs = np.array(coefs, dtype="int64")
                print("係数 :",coefs)
                buf.append(f"係数 : {coefs}")
                print("t =",v)
                buf.append(f"t = {v}")
                print("復号結果：", np.dot(coefs, v) % m)
                buf.append(f"復号結果 : {np.dot(coefs, v) % m}")
        return "\n".join(buf)
            
    
class SecretSharingScheme(object):
    '''
    秘密分散共有法のオブジェクト
    '''
    def __init__(self, matrix, modulo, participants):
        if matrix.shape[1] - 1 < len(participants):
            raise ValueError("Not Matching Size")
        self.modulo = modulo
        self.m = modulo
        self.nonunits = {i for i in range(2, self.m) if np.gcd(i, modulo) != 1} #最大公約数が1ではないもので回す（非可逆元の生成）
        self.matrix = matrix
        self.G = matrix
        self.participants = participants
        self.P = participants
        self.converter = lambda x: convertFromCoreNumbersMod(x , modulo)
        self.cyphertext = None

    def is_decodable(self, set_ps):
        '''
        与えられた参加者の部分集合が秘密を復号可能かどうかを判定し,\n
        結果の真偽値とベクトルの線形結合における係数のタプルを返します
        '''
        matrix = self.G
        m = self.m
        nonunits = self.nonunits
        converter = self.converter
        succeed = False
        shares = [(p.share, p.label) for p in set_ps]
        symbols = " ".join("abcdefghijklmnopqrstuvwxyz"[:len(shares)])
        xs = sym.symbols(symbols) if len(shares) >= 2 else [sym.symbols('a')] #変数が一個の時201行目のfor分でおかしくなるため今回はリストに入れてる
        A = np.array([(matrix.T[share[1]]) for share in shares]).T  #連立方程式の係数行列
        print(A)
        equations = []
        for row,j in zip(A , matrix.T[0]): #j:生成行列の転置の一列目を持ってくる, row:Aの行
            equation = -j # 初期値として1を移項しておきたいため-jとしている.
            for i,x in zip(row , xs): #rowの行ベクトルの成分でループ 
                equation += i*x
            equations.append(equation) #=0でときたいものをリストに加えている.
        solution = sym.solve(equations)
        coefs = []
        for x in xs:
            try:
                coefs.append(solution[x])
            except KeyError:  
                coefs.append(x) #変数の数が辞書の結びつけを行っているものよりも少なくなる時がある.
            except TypeError:
                return (False , None) #解なしの時dict型で返さない時の処理
        sample_coefs = coefs
        for x in xs:
            try:
                solution[x]
            except KeyError:
                pseudoinv = lambda x: sym.Rational(1, x)
                pseudoinvs = [pseudoinv(i) for i in nonunits ] 
                for i in range(m):
                    sample_coefs = [f.subs(x , i) for f in sample_coefs ] #xにiを代入
                    sample_coefs = [converter(x) for x in sample_coefs]  
                    if None not in sample_coefs:
                        succeed = True
                        break
                    sample_coefs = coefs
                if not succeed:
                    for i in pseudoinvs:
                        sample_coefs = [f.subs(x , i) for f in sample_coefs ]
                        sample_coefs =[converter(x) for x in sample_coefs] 
                        if None not in sample_coefs:
                            succeed = True
                            break
                        sample_coefs = coefs
        sample_coefs =[converter(x) for x in sample_coefs] 
        if None not in sample_coefs:
            succeed = True
        if not succeed:
            return(False, None)
        if None in sample_coefs:
            return (False, None)
        return (True , sample_coefs)


def input_row(column_num, index=None):
    '''
    半角スペース区切りで整数列の入力を受け取り,numpy.array型に変換します\n
    指定した要素数(列数)に満たない場合は入力をやり直します\n
    インデックス指定で行のインデックスを表示します
    '''
    if index is None:
        s = input("ベクトルを入力してください : ")
    else:
        s = input(f"{index}行目 : ")
    row = list(map(int, s.split()))
    if len(row) == column_num:
        return row
    else:
        return input_row(column_num, index)        


def write_file(file_name: str, buf: list):
    '''
    ファイル名とリストを入れたらファイルに出力します。
    '''
    with open(file_name, "w", encoding='utf-8') as f:
        f.write("\n".join([str(b) for b in buf]))


def main():
    m = 4 #65519
    n = int(input("列数を入力してください:"))
    k = int(input("行数を入力してください:"))
    print("生成行列を入力してください")
    buf = []
    G = np.array([input_row(n, i+1) for i in range(k)])
    print("G=", G)
    buf.append(G)
    P = [Participant(input(f"参加者{i + 1} : ")) for i in range(n-1)]
    SSS = SecretSharingScheme(G, m , P)
    dealer = Dealer(input("ディーラー : "),SSS)
    dealer.generate_u()
    print("s=",dealer.secret)
    buf.append(f"s = {dealer.secret}")
    print("u=",dealer.u)
    buf.append(f"u = {dealer.u}")
    dealer.generate_shares()
    print("t=",dealer.t)
    buf.append(f"t = {dealer.t}")
    dealer.randomly_distribute_shares()
    print(SSS.is_decodable(P))
    buf.append(SSS.is_decodable(P))
    buf.append(dealer.check_decodable())
    

    write_file("result.txt", buf)



if __name__ == "__main__":
   main() 
