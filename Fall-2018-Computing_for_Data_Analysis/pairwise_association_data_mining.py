
###############
# Exercise 0
##############
def eval_strint(s, base=2):
    assert type(s) is str
    assert 2 <= base <= 36
    
    # convert string to integer given the input base
    return(int(s, base))

eval_strint('100111010', 2)
    
###############
# Exercise 1
##############

def is_valid_strfrac(s, base=2):
    return all([is_valid_strdigit(c, base) for c in s if c != '.']) \
        and (len([c for c in s if c == '.']) <= 1)
    
def eval_strfrac(s, base=2):
    assert is_valid_strfrac(s, base), "'{}' contains invalid digits for a base-{} number.".format(s, base)
    
    if '.' in s:
        # Convert string to list
        str_list = list(s)
        
        # Loop through the string of values
        total = 0
        index = 0
        for i in str_list:
            
            # Create index based on position of decimal
            if index < str_list.index('.'):
                i_index = str_list.index('.') - index - 1
            elif index > str_list.index('.'):
                i_index = str_list.index('.') - index
            
            # Calculate digit value based on position relative to decimal point
            if i != '.':
                total += int(i, base) * base**i_index
            index += 1
        return(total)
    else:
        return(float(int(s, base)))

eval_strfrac('3.14', 10)
eval_strfrac('100.101', 2)
eval_strfrac('2c', 16)
eval_strfrac('f.a', 16)

###############
# Exercise 2
##############

import struct
import array

def fp_bin(v):
    assert type(v) is float
    
    # Convert float value to hex
    v = v.hex()
    
    # Extract sign bit (- or +)
    if v[0] == '-':
        s_sign = '-'
    else:
        s_sign = '+'
    
    # Get location of x and p to store significand and exponent
    x_index = v.find('x')
    p_index = v.find('p')
    
    # Extract significand using eval_strfrac function and convert to 54 bits
    i_signif = v[x_index + 1:p_index]
    i_signif = eval_strfrac(i_signif, base=16)
    
    #bin_signif = ''.join(bin(struct.unpack('B', c[0])[0]).replace('0b', '').zfill(8) for c in struct.pack('!d', i_signif))
    packed = struct.pack('!d', i_signif)
    integers = array.array('B', packed)
    binaries = [bin(i) for i in integers]
    stripped_binaries = [s.replace('0b', '') for s in binaries]
    padded = [s.rjust(8, '0') for s in stripped_binaries]
    s_signif = ''.join(padded)

    s_signif = s_signif[11:12] + '.' + s_signif[12:]
    
    # Extract exponent as integer
    v_exp = int(v[p_index + 1:])
    
    # Store value as tuple
    return(s_sign, s_signif, v_exp)



fp_bin(-0.1)
check_fp_bin(-0.1, ('-', '1.1001100110011001100110011001100110011001100110011010', -4))


###############
# Exercise 3
##############


from ast import literal_eval

def eval_fp(sign, significand, exponent, base=2):
    assert sign in ['+', '-'], "Sign bit must be '+' or '-', not '{}'.".format(sign)
    assert is_valid_strfrac(significand, base), "Invalid significand for base-{}: '{}'".format(base, significand)
    assert type(exponent) is int
        
    if sign == '-':
        return(-eval_strfrac(significand, base) * base**exponent)
    else:
        return(eval_strfrac(significand, base) * base**exponent)




eval_fp('+', '1.25000', -1, 10) # 0.125

eval_fp('+', '1.1011001001100111100010001011001000100000111110010000', -5)




###############
# Exercise 4
##############

def add_fp_bin(u, v, signif_bits):
    u_sign, u_signif, u_exp = u
    v_sign, v_signif, v_exp = v
    
    # You may assume normalized inputs at the given precision, `signif_bits`.
    assert u_signif[:2] == '1.' and len(u_signif) == (signif_bits+1)
    assert v_signif[:2] == '1.' and len(v_signif) == (signif_bits+1)
    
    # Convert u and v to floating point values
    u = eval_fp(u_sign, u_signif, u_exp)
    v = eval_fp(v_sign, v_signif, v_exp)
    
    # Add u and v
    t = u + v
    
    # Convert t to tuple
    t = fp_bin(t)
    
    # Trim significand by signif_bits
    t_sign, t_signif, t_exp = t
    t_signif = t_signif[:signif_bits+1]
    assert t_signif[:2] == '1.' and len(t_signif) == (signif_bits+1)
    
    t = (t_sign, t_signif, t_exp)
    
    return(t)
    
    
    
u = ('+', '1.010010', 0)
v = ('-', '1.000000', -2)
signif_bits = 7
add_fp_bin(u,v,signif_bits)

w_true = ('+', '1.000010', 0)


u = ('+', '1.00000', 0)
v = ('+', '1.00000', -5)
signif_bits = 6
add_fp_bin(u, v, signif_bits)
w_true = ('+', '1.00001', 0)



u = ('+', '1.00000', 0)
v = ('-', '1.00000', -5)
signif_bits = 6
add_fp_bin(u,v,signif_bits)

w_true = ('+', '1.11110', -1)


u = ('+', '1.00000', 0)
v = ('+', '1.00000', -6)
signif_bits = 6
add_fp_bin(u,v,signif_bits)

w_true = ('+', '1.00000', 0)


u = ('+', '1.00000', 0)
v = ('-', '1.00000', -6)
signif_bits = 6
add_fp_bin(u,v,signif_bits)

w_true = ('+', '1.11111', -1)






