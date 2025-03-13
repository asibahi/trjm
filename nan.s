	.globl  _main
	.text
_main:
	pushq   %rbp
	movq    %rsp,   %rbp
	subq    $256,   %rsp
	movsd   _L_c_133(%rip), %xmm15
	movsd   %xmm15, -8(%rbp)
	movsd   _zero.1(%rip), %xmm15
	movsd   -8(%rbp), %xmm14
	divsd   %xmm15, %xmm14
	movsd   %xmm14, -8(%rbp)
	movsd   -8(%rbp), %xmm15
	movsd   %xmm15, -16(%rbp)
	movsd   -16(%rbp), %xmm15
	comisd  _L_c_133(%rip), %xmm15
	movl    $0,     -20(%rbp)
	setp    -20(%rbp)
	movl    -20(%rbp), %r11d
	notl    %r11d
	movl    $0,     -20(%rbp)
	setb    -20(%rbp)
	andl    %r11d,  -20(%rbp)
	cmpl    $0,     -20(%rbp)
	jne     .Ljmp.70
	movsd   -16(%rbp), %xmm15
	comisd  _L_c_133(%rip), %xmm15
	movl    $0,     -24(%rbp)
	setp    -24(%rbp)
	movl    -24(%rbp), %r11d
	notl    %r11d
	movl    $0,     -24(%rbp)
	sete    -24(%rbp)
	andl    %r11d,  -24(%rbp)
	cmpl    $0,     -24(%rbp)
	jne     .Ljmp.70
	movl    $0,     -28(%rbp)
	jmp     .Lend.70
  .Ljmp.70:
	movl    $1,     -28(%rbp)
  .Lend.70:
	cmpl    $0,     -28(%rbp)
	jne     .Ljmp.68
	movsd   -16(%rbp), %xmm15
	comisd  _L_c_133(%rip), %xmm15
	movl    $0,     -32(%rbp)
	setp    -32(%rbp)
	movl    -32(%rbp), %r11d
	notl    %r11d
	movl    $0,     -32(%rbp)
	seta    -32(%rbp)
	andl    %r11d,  -32(%rbp)
	cmpl    $0,     -32(%rbp)
	jne     .Ljmp.68
	movl    $0,     -36(%rbp)
	jmp     .Lend.68
  .Ljmp.68:
	movl    $1,     -36(%rbp)
  .Lend.68:
	cmpl    $0,     -36(%rbp)
	jne     .Ljmp.66
	movsd   -16(%rbp), %xmm15
	comisd  _L_c_133(%rip), %xmm15
	movl    $0,     -40(%rbp)
	setp    -40(%rbp)
	movl    -40(%rbp), %r11d
	notl    %r11d
	movl    $0,     -40(%rbp)
	setbe   -40(%rbp)
	andl    %r11d,  -40(%rbp)
	cmpl    $0,     -40(%rbp)
	jne     .Ljmp.66
	movl    $0,     -44(%rbp)
	jmp     .Lend.66
  .Ljmp.66:
	movl    $1,     -44(%rbp)
  .Lend.66:
	cmpl    $0,     -44(%rbp)
	jne     .Ljmp.64
	movsd   -16(%rbp), %xmm15
	comisd  _L_c_133(%rip), %xmm15
	movl    $0,     -48(%rbp)
	setp    -48(%rbp)
	movl    -48(%rbp), %r11d
	notl    %r11d
	movl    $0,     -48(%rbp)
	setae   -48(%rbp)
	andl    %r11d,  -48(%rbp)
	cmpl    $0,     -48(%rbp)
	jne     .Ljmp.64
	movl    $0,     -52(%rbp)
	jmp     .Lend.64
  .Ljmp.64:
	movl    $1,     -52(%rbp)
  .Lend.64:
	cmpl    $0,     -52(%rbp)
	je      .Lelse.63
	movl    $1,     %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
  .Lelse.63:
	movl    $1,     %r10d
	cvtsi2sdl %r10d, %xmm15
	movsd   %xmm15, -64(%rbp)
	movsd   -64(%rbp), %xmm15
	comisd  -16(%rbp), %xmm15
	movl    $0,     -68(%rbp)
	setp    -68(%rbp)
	movl    -68(%rbp), %r11d
	notl    %r11d
	movl    $0,     -68(%rbp)
	setb    -68(%rbp)
	andl    %r11d,  -68(%rbp)
	cmpl    $0,     -68(%rbp)
	jne     .Ljmp.84
	movl    $1,     %r10d
	cvtsi2sdl %r10d, %xmm15
	movsd   %xmm15, -80(%rbp)
	movsd   -80(%rbp), %xmm15
	comisd  -16(%rbp), %xmm15
	movl    $0,     -84(%rbp)
	setp    -84(%rbp)
	movl    -84(%rbp), %r11d
	notl    %r11d
	movl    $0,     -84(%rbp)
	sete    -84(%rbp)
	andl    %r11d,  -84(%rbp)
	cmpl    $0,     -84(%rbp)
	jne     .Ljmp.84
	movl    $0,     -88(%rbp)
	jmp     .Lend.84
  .Ljmp.84:
	movl    $1,     -88(%rbp)
  .Lend.84:
	cmpl    $0,     -88(%rbp)
	jne     .Ljmp.82
	movl    $1,     %r10d
	cvtsi2sdl %r10d, %xmm15
	movsd   %xmm15, -96(%rbp)
	movsd   -96(%rbp), %xmm15
	comisd  -16(%rbp), %xmm15
	movl    $0,     -100(%rbp)
	setp    -100(%rbp)
	movl    -100(%rbp), %r11d
	notl    %r11d
	movl    $0,     -100(%rbp)
	seta    -100(%rbp)
	andl    %r11d,  -100(%rbp)
	cmpl    $0,     -100(%rbp)
	jne     .Ljmp.82
	movl    $0,     -104(%rbp)
	jmp     .Lend.82
  .Ljmp.82:
	movl    $1,     -104(%rbp)
  .Lend.82:
	cmpl    $0,     -104(%rbp)
	jne     .Ljmp.80
	movl    $1,     %r10d
	cvtsi2sdl %r10d, %xmm15
	movsd   %xmm15, -112(%rbp)
	movsd   -112(%rbp), %xmm15
	comisd  -16(%rbp), %xmm15
	movl    $0,     -116(%rbp)
	setp    -116(%rbp)
	movl    -116(%rbp), %r11d
	notl    %r11d
	movl    $0,     -116(%rbp)
	setbe   -116(%rbp)
	andl    %r11d,  -116(%rbp)
	cmpl    $0,     -116(%rbp)
	jne     .Ljmp.80
	movl    $0,     -120(%rbp)
	jmp     .Lend.80
  .Ljmp.80:
	movl    $1,     -120(%rbp)
  .Lend.80:
	cmpl    $0,     -120(%rbp)
	jne     .Ljmp.78
	movl    $1,     %r10d
	cvtsi2sdl %r10d, %xmm15
	movsd   %xmm15, -128(%rbp)
	movsd   -128(%rbp), %xmm15
	comisd  -16(%rbp), %xmm15
	movl    $0,     -132(%rbp)
	setp    -132(%rbp)
	movl    -132(%rbp), %r11d
	notl    %r11d
	movl    $0,     -132(%rbp)
	setae   -132(%rbp)
	andl    %r11d,  -132(%rbp)
	cmpl    $0,     -132(%rbp)
	jne     .Ljmp.78
	movl    $0,     -136(%rbp)
	jmp     .Lend.78
  .Ljmp.78:
	movl    $1,     -136(%rbp)
  .Lend.78:
	cmpl    $0,     -136(%rbp)
	je      .Lelse.77
	movl    $2,     %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
  .Lelse.77:
	movsd   -16(%rbp), %xmm15
	comisd  -16(%rbp), %xmm15
	movl    $0,     -140(%rbp)
	setp    -140(%rbp)
	movl    -140(%rbp), %r11d
	notl    %r11d
	movl    $0,     -140(%rbp)
	sete    -140(%rbp)
	andl    %r11d,  -140(%rbp)
	cmpl    $0,     -140(%rbp)
	je      .Lelse.96
	movl    $3,     %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
  .Lelse.96:
	movsd   -16(%rbp), %xmm15
	comisd  -16(%rbp), %xmm15
	movl    $0,     -144(%rbp)
	setp    -144(%rbp)
	movl    -144(%rbp), %r11d
	movl    $0,     -144(%rbp)
	setne   -144(%rbp)
	orl     %r11d,  -144(%rbp)
	cmpl    $0,     -144(%rbp)
	movl    $0,     -148(%rbp)
	sete    -148(%rbp)
	cmpl    $0,     -148(%rbp)
	je      .Lelse.98
	movl    $4,     %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
  .Lelse.98:
	movsd   -16(%rbp), %xmm0
	call    _double_isnan
	movl    %eax,   -152(%rbp)
	cmpl    $0,     -152(%rbp)
	movl    $0,     -156(%rbp)
	sete    -156(%rbp)
	cmpl    $0,     -156(%rbp)
	je      .Lelse.101
	movl    $5,     %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
  .Lelse.101:
	movl    $4,     %r10d
	cvtsi2sdl %r10d, %xmm15
	movsd   %xmm15, -168(%rbp)
	movsd   -168(%rbp), %xmm15
	movsd   %xmm15, -176(%rbp)
	movsd   -176(%rbp), %xmm14
	mulsd   -16(%rbp), %xmm14
	movsd   %xmm14, -176(%rbp)
	movsd   -176(%rbp), %xmm0
	call    _double_isnan
	movl    %eax,   -180(%rbp)
	cmpl    $0,     -180(%rbp)
	movl    $0,     -184(%rbp)
	sete    -184(%rbp)
	cmpl    $0,     -184(%rbp)
	je      .Lelse.104
	movl    $6,     %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
  .Lelse.104:
	movsd   _L_c_134(%rip), %xmm15
	movsd   %xmm15, -192(%rbp)
	movsd   -16(%rbp), %xmm15
	movsd   -192(%rbp), %xmm14
	divsd   %xmm15, %xmm14
	movsd   %xmm14, -192(%rbp)
	movsd   -192(%rbp), %xmm0
	call    _double_isnan
	movl    %eax,   -196(%rbp)
	cmpl    $0,     -196(%rbp)
	movl    $0,     -200(%rbp)
	sete    -200(%rbp)
	cmpl    $0,     -200(%rbp)
	je      .Lelse.109
	movl    $7,     %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
  .Lelse.109:
	movsd   -16(%rbp), %xmm15
	movsd   %xmm15, -208(%rbp)
	movsd   _L_neg0(%rip), %xmm15
	movsd   -208(%rbp), %xmm14
	xorpd   %xmm15, %xmm14
	movsd   %xmm14, -208(%rbp)
	movsd   -208(%rbp), %xmm0
	call    _double_isnan
	movl    %eax,   -212(%rbp)
	cmpl    $0,     -212(%rbp)
	movl    $0,     -216(%rbp)
	sete    -216(%rbp)
	cmpl    $0,     -216(%rbp)
	je      .Lelse.113
	movl    $8,     %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
  .Lelse.113:
	xorpd   %xmm14, %xmm14
	comisd  -16(%rbp), %xmm14
	movl    $0,     -220(%rbp) ; zero out rbp
	sete    -220(%rbp)	
	cmpl    $0,     -220(%rbp)
	je      .Lelse.117
	movl    $9,     %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
  .Lelse.117:
	xorpd   %xmm15, %xmm15
	comisd  -16(%rbp), %xmm15
	je      .Lelse.119
	jmp     .Lend.119
  .Lelse.119:
	movl    $10,    %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
  .Lend.119:
	movl    $0,     -224(%rbp)
  .Lstrt.for.32:
	xorpd   %xmm15, %xmm15
	comisd  -16(%rbp), %xmm15
	je      .Lbrk.for.32
	movl    $1,     -224(%rbp)
	jmp     .Lbrk.for.32
  .Lcntn.for.32:
	jmp     .Lstrt.for.32
  .Lbrk.for.32:
	cmpl    $0,     -224(%rbp)
	movl    $0,     -228(%rbp)
	sete    -228(%rbp)
	cmpl    $0,     -228(%rbp)
	je      .Lelse.120
	movl    $11,    %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
  .Lelse.120:
	movl    $0,     -224(%rbp)
  .Lcntn.while.40:
	xorpd   %xmm15, %xmm15
	comisd  -16(%rbp), %xmm15
	je      .Lbrk.while.40
	movl    $1,     -224(%rbp)
	jmp     .Lbrk.while.40
	jmp     .Lcntn.while.40
  .Lbrk.while.40:
	cmpl    $0,     -224(%rbp)
	movl    $0,     -232(%rbp)
	sete    -232(%rbp)
	cmpl    $0,     -232(%rbp)
	je      .Lelse.122
	movl    $12,    %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
  .Lelse.122:
	movl    $1,     -236(%rbp)
	negl    -236(%rbp)
	movl    -236(%rbp), %r10d
	movl    %r10d,  -224(%rbp)
  .Lstrt.do.48:
	movl    -224(%rbp), %r10d
	movl    %r10d,  -240(%rbp)
	addl    $1,     -240(%rbp)
	movl    -240(%rbp), %r10d
	movl    %r10d,  -224(%rbp)
	cmpl    $0,     -224(%rbp)
	je      .Lelse.126
	jmp     .Lbrk.do.48
  .Lelse.126:
  .Lcntn.do.48:
	xorpd   %xmm15, %xmm15
	comisd  -16(%rbp), %xmm15
	jne     .Lstrt.do.48
  .Lbrk.do.48:
	cmpl    $0,     -224(%rbp)
	movl    $0,     -244(%rbp)
	sete    -244(%rbp)
	cmpl    $0,     -244(%rbp)
	je      .Lelse.127
	movl    $13,    %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
  .Lelse.127:
	xorpd   %xmm15, %xmm15
	comisd  -16(%rbp), %xmm15
	je      .Le2.129
	movl    $1,     -248(%rbp)
	jmp     .Lend.129
  .Le2.129:
	movl    $0,     -248(%rbp)
  .Lend.129:
	movl    -248(%rbp), %r10d
	movl    %r10d,  -224(%rbp)
	cmpl    $0,     -224(%rbp)
	movl    $0,     -252(%rbp)
	sete    -252(%rbp)
	cmpl    $0,     -252(%rbp)
	je      .Lelse.131
	movl    $14,    %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
  .Lelse.131:
	xorl    %eax,   %eax
	movq    %rbp,   %rsp
	popq    %rbp
	ret
	.data
	.balign 8
_zero.1:
	.quad   0
	.literal16
	.balign 16
_L_neg0:
	.quad   9223372036854775808
	.quad   0
	.literal8
	.balign 8
_L_upper_bound:
	.quad   4890909195324358656
	.literal8
	.balign 8
_L_c_133:
	.quad   0
	.literal8
	.balign 8
_L_c_134:
	.quad   4657056266235936768
