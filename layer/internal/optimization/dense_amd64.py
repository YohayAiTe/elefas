from peachpy import *
from peachpy.x86_64 import *
from peachpy.x86_64 import uarch, isa

input_ptr = Argument(ptr(const_float_), name="input_ptr")
kernel_ptr = Argument(ptr(const_float_), name="kernel_ptr")
bias_ptr = Argument(ptr(const_float_), name="bias_ptr")
output_ptr = Argument(ptr(const_float_), name="output_ptr")
batch_count = Argument(const_int64_t, name="batch_count")
input_units = Argument(const_int64_t, name="input_units")
output_units = Argument(const_int64_t, name="output_units")

arguments = (input_ptr, kernel_ptr, bias_ptr, output_ptr, batch_count, input_units, output_units)

with Function("DenseApplyF32", arguments, target=uarch.default + isa.fma3):
    reg_input_ptr = GeneralPurposeRegister64()
    reg_base_input_ptr = GeneralPurposeRegister64()
    reg_kernel_ptr = GeneralPurposeRegister64()
    reg_base_kernel_ptr = GeneralPurposeRegister64()
    reg_bias_ptr = GeneralPurposeRegister64()
    reg_base_bias_ptr = GeneralPurposeRegister64()
    reg_output_ptr = GeneralPurposeRegister64()
    # reg_base_output_ptr = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_base_input_ptr, input_ptr)
    LOAD.ARGUMENT(reg_base_kernel_ptr, kernel_ptr)
    LOAD.ARGUMENT(reg_base_bias_ptr, bias_ptr)
    LOAD.ARGUMENT(reg_output_ptr, output_ptr)

    reg_batch_count = GeneralPurposeRegister64()
    reg_input_units = GeneralPurposeRegister64()
    reg_output_units = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_batch_count, batch_count)
    LOAD.ARGUMENT(reg_input_units, input_units)
    LOAD.ARGUMENT(reg_output_units, output_units)

    with Loop() as batch_loop:
        output_index = GeneralPurposeRegister64()
        MOV(output_index, 0)
        MOV(reg_kernel_ptr, reg_base_kernel_ptr)
        MOV(reg_bias_ptr, reg_base_bias_ptr)

        with Loop() as output_loop:
            MOV(reg_input_ptr, reg_base_input_ptr)
            input_index = GeneralPurposeRegister64()
            MOV(input_index, 0)

            unroll_factor = 4
            vector_input_loop = Loop()
            scalar_input_loop = Loop()
            ymm_accs = [YMMRegister() for _ in range(unroll_factor)]
            for ymm_acc in ymm_accs:
                VXORPS(ymm_acc, ymm_acc, ymm_acc)

            ADD(input_index, 8*unroll_factor)
            CMP(reg_input_units, input_index)
            JA(vector_input_loop.end)
            with vector_input_loop:
                ymm_xs = [YMMRegister() for _ in range(unroll_factor)]
                for (i, ymm_x) in enumerate(ymm_xs):
                    VMOVUPS(ymm_x, [reg_input_ptr + 32*i])
                for (i, (ymm_acc, ymm_x)) in enumerate(zip(ymm_accs, ymm_xs)):
                    VFMADD231PS(ymm_acc, ymm_x, [reg_kernel_ptr + 32*i])

                # next iteration
                ADD(reg_input_ptr, 32*unroll_factor)
                ADD(reg_kernel_ptr, 32*unroll_factor)
                ADD(input_index, 8*unroll_factor)
                CMP(reg_input_units, input_index)
                JBE(vector_input_loop.begin)
            
            # Reduction of multiple YMM registers into into YMM register
            VADDPS(ymm_accs[0], ymm_accs[0], ymm_accs[1])
            VADDPS(ymm_accs[2], ymm_accs[2], ymm_accs[3])
            VADDPS(ymm_accs[0], ymm_accs[0], ymm_accs[2])
            
            xmm_acc = ymm_accs[0].as_xmm
            xmm_tmp = XMMRegister()
            VEXTRACTF128(xmm_tmp, ymm_accs[0], 1)
            VADDPS(xmm_acc, xmm_acc, xmm_tmp)
            VHADDPS(xmm_acc, xmm_acc, xmm_acc)
            VHADDPS(xmm_acc, xmm_acc, xmm_acc)
            VADDSS(xmm_acc, xmm_acc, [reg_bias_ptr])

            SUB(input_index, 8*unroll_factor)
            CMP(input_index, reg_input_units)
            JZ(scalar_input_loop.end)
            with scalar_input_loop:
                temp = XMMRegister()
                VMOVSS(temp, [reg_input_ptr])
                VFMADD231SS(xmm_acc, temp, [reg_kernel_ptr])

                # next iteration
                ADD(reg_kernel_ptr, 4)
                ADD(reg_input_ptr, 4)
                INC(input_index)
                CMP(input_index, reg_input_units)
                JNZ(scalar_input_loop.begin)
            
            VMOVSS([reg_output_ptr], xmm_acc)
            # next iteration
            ADD(reg_bias_ptr, 4)
            ADD(reg_output_ptr, 4)
            INC(output_index)
            CMP(output_index, reg_output_units)
            JNZ(output_loop.begin)
        
        # next iteration
        ADD(reg_base_input_ptr, reg_input_units)
        ADD(reg_base_input_ptr, reg_input_units)
        ADD(reg_base_input_ptr, reg_input_units)
        ADD(reg_base_input_ptr, reg_input_units)
        DEC(reg_batch_count)
        JNZ(batch_loop.begin)
    
    RETURN()
