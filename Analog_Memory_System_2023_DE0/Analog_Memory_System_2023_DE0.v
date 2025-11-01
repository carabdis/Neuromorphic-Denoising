module Analog_Memory_System_2023_DE0

#(
	// Parameter Declarations
	parameter ROW_BIT = 4,
	parameter COL_BIT = 3,
	parameter GROUP = 4,
	parameter BL_MODE = 3, 
	parameter SL_MODE = 1,
	parameter WL_MODE = 2,
	parameter COL_START = 0,
	parameter ROW_START = 0,
	parameter SET_RECOVER_TIME = 10,
	parameter RESET_RECOVER_TIME = 10000,
	parameter RESET_STABLIZE_TIME = 20000,
	parameter PARA_COMPUTE_TIME = 65535,
	parameter DECLINE = 1040,
	parameter RESET_TIME = 10000,
	parameter READ_TIME = 30000,
	parameter NOLAP_TIME = 100,
	parameter DISCHARGE_TIME = 20000,
	parameter INITIAL = 1000,
	parameter STANDARD = 40000,
	parameter STEP = 4096, //625
	parameter TEST_STEP = 65,
	parameter INT_BIAS = 0,
	parameter SAMPLE_TIME = 5500,
	parameter START_POINT = 4000
)

(
	// Input Ports
	input wire CLK_IN,
	input wire RST,
	input wire [2:0] MODE,
	input wire EN,
	input wire WL_SELECT,
//	input wire [ROW_BIT - 1:0] WL,
//	input wire [COL_BIT - 1:0] BL,
	input wire DEBUG,
	input wire DEBUG_IN,
	input wire [3:0] COUNT_PORT_SIDE,
	input wire isck,
	inout wire sda,

	// Output Ports
	output wire EN_PCB,
	output reg [ROW_BIT - 1:0] WL_CTRL,
	output reg [COL_BIT - 1:0] BL_CTRL,
	output reg [ROW_BIT - 1:0] WL_CTRL_SUB,
	output reg [COL_BIT - 1:0] BL_CTRL_SUB,
	output reg [GROUP - 1:0] COL_EN,
	output reg [GROUP - 1:0] ROW_EN,
	output reg [GROUP - 1:0] COL_EN_SUB,
	output reg [GROUP - 1:0] ROW_EN_SUB,
	output reg [BL_MODE - 1:0] BL_BUS_CHS,
	output reg [WL_MODE - 1:0] WL_BUS_CHS,
	output reg [SL_MODE - 1:0] SL_BUS_CHS,
	output reg [BL_MODE - 1:0] BL_BUS_CHS_SUB,
	output reg [WL_MODE - 1:0] WL_BUS_CHS_SUB,
	output reg [SL_MODE - 1:0] SL_BUS_CHS_SUB,
	output reg Sense_Ctrl,
	output reg Reset,
	output reg CONN,
	output reg CAP_CTRL,
	output reg GEN_CTRL,
	output reg WHIGH_CTRL,
	output reg WLOW_CTRL,
	output reg FPGA_ENABLE,
	output reg Sense_Ctrl_SUB,
	output reg Reset_SUB,
	output reg CONN_SUB,
	output reg CAP_CTRL_SUB,
	output reg GEN_CTRL_SUB,
	output reg WHIGH_CTRL_SUB,
	output reg WLOW_CTRL_SUB,
	output reg FPGA_ENABLE_SUB,
	output wire [6:0] HEX_MODE,
	output wire [6:0] HEX_PFSM,
	output reg DEBUG_OUT
//	output wire o_sda,
);
	reg [31:0] COUNT;
	reg [31:0] DEBUG_COUNT;
	reg [1:0] COMMAND;
	reg [2:0] PULSE_MACHINE;
	reg [31:0] FIRST_COUNT;
	reg [31:0] OUT_COUNT;
	wire [3:0] Para_Compute [3:0];
	reg [3:0] Para_Count;
	reg Target;
	wire Symbol_;
	reg Symbol;
	wire Symbol_SUB;
	wire CLK;
	wire LOCKED;
	wire rstn;
	wire [7:0] ENABLE_BYTE;
	wire [7:0] ADDR_BYTE;
	wire [7:0] ENABLE_BYTE_SUB;
	wire [7:0] ADDR_BYTE_SUB;
	reg [7:0] CHECK_BYTE;
	wire [7:0] PARA_BYTE [1:0];
	wire [ROW_BIT - 1:0] WL;
	wire [COL_BIT - 1:0] BL;
	wire [ROW_BIT - 1:0] WL_SUB;
	wire [COL_BIT - 1:0] BL_SUB;
	wire [GROUP - 1:0] ROW_EN_;
	wire [GROUP - 1:0] COL_EN_;
	wire [GROUP - 1:0] ROW_EN_SUB_;
	wire [GROUP - 1:0] COL_EN_SUB_;
	wire [GROUP - 1:0] JUDGE_COMPUTE;
	wire [7:0] WRITING_VALUE;
	wire COUNT_PORT;
	wire COMPUTE;
	wire [7:0] MODE_BYTE;
	wire [2:0] MODE_;
	assign EN_PCB = EN;
	assign rstn = ~RST;
	assign COUNT_PORT = | (COL_EN & ~COUNT_PORT_SIDE);
	assign MODE_ = DEBUG ? MODE: MODE_BYTE[2:0];
//	assign DEBUG_OUT = JUDGE_COMPUTE[0];
	PLL_0002 PLL_TEST (
		.refclk(CLK_IN),
		.rst(RST),
		.outclk_0(CLK),
		.locked(LOCKED)
	);
	
	bufif0  ( sda, 1'b0, osda );	// Slave's SDA IO
	i2c_slave i2c (
		.clk(CLK_IN),
		.rstn(rstn),
		.I_DEV_ADR(7'h6C),
		.isda(sda),
		.osda(osda),
		.isck(isck),
		// write into chip
		.reg00d(ENABLE_BYTE),
		.reg01d(ENABLE_BYTE_SUB),
		.reg02d(MODE_BYTE),
		.reg03d(),
		.reg04d(), 
		.reg05d(),
		.reg06d(ADDR_BYTE),
		.reg07d(ADDR_BYTE_SUB),
		.reg08d(WRITING_VALUE),
		.reg09d(PARA_BYTE[1]),
		.reg0ad(PARA_BYTE[0]),
		.reg0bd(),
		//read from chip
		.ireg02d(OUT_COUNT[7:0]),
		.ireg03d(OUT_COUNT[15:8]),
		.ireg04d(OUT_COUNT[23:16]),
		.ireg05d(OUT_COUNT[31:24]),
		.ireg08d(FIRST_COUNT[7:0]),
		.ireg09d(FIRST_COUNT[15:8]),
		.ireg0ad(FIRST_COUNT[23:16]),
		.ireg0bd(FIRST_COUNT[31:24]),
		.ireg07d(CHECK_BYTE)
	);
	assign Para_Compute[3] = PARA_BYTE[1][7:4];
	assign Para_Compute[2] = PARA_BYTE[1][3:0];
	assign Para_Compute[1] = PARA_BYTE[0][7:4];
	assign Para_Compute[0] = PARA_BYTE[0][3:0];
	assign JUDGE_COMPUTE[0] = (Para_Compute[0] > Para_Count) ? 1'b1:((MODE_==3'b010) ? OUT_COUNT < INT_BIAS:1'b0);
	assign JUDGE_COMPUTE[1] = (Para_Compute[1] > Para_Count) ? 1'b1:((MODE_==3'b010) ? OUT_COUNT < INT_BIAS:1'b0);
	assign JUDGE_COMPUTE[2] = (Para_Compute[2] > Para_Count) ? 1'b1:((MODE_==3'b010) ? OUT_COUNT < INT_BIAS:1'b0);
	assign JUDGE_COMPUTE[3] = (Para_Compute[3] > Para_Count) ? 1'b1:((MODE_==3'b010) ? OUT_COUNT < INT_BIAS:1'b0);
	assign COMPUTE = & COUNT[11:0];
	assign ROW_EN_ = ENABLE_BYTE[7:4] & JUDGE_COMPUTE;
	assign COL_EN_ = ENABLE_BYTE[3:0];
	assign WL = ADDR_BYTE[7:4];
	assign BL = ADDR_BYTE[3:1];
	assign ROW_EN_SUB_ = ENABLE_BYTE_SUB[7:4];
	assign COL_EN_SUB_ = ENABLE_BYTE_SUB[3:0];
	assign WL_SUB = ADDR_BYTE_SUB[7:4];
	assign BL_SUB = ADDR_BYTE_SUB[3:1];
	assign Symbol_ = ADDR_BYTE[0];
	assign Symbol_SUB = ADDR_BYTE_SUB[0];
	assign HEX_MODE[0] = ~((MODE_[1] & MODE_[0]) | (MODE_[1] & (~MODE_[0])) | ((!MODE_[1]) & (!MODE_[0])));
	assign HEX_MODE[1] = ~((MODE_[1] & MODE_[0]) | (MODE_[1] & (~MODE_[0])) | ((!MODE_[1]) & MODE_[0]) |
								((!MODE_[1]) & (!MODE_[0])));
	assign HEX_MODE[2] = ~((MODE_[1] & MODE_[0]) | ((!MODE_[1]) & MODE_[0]) | ((!MODE_[1]) & (!MODE_[0])));
	assign HEX_MODE[3] = ~((MODE_[1] & MODE_[0]) | (MODE_[1] & (~MODE_[0])) | ((!MODE_[1]) & (!MODE_[0])));
	assign HEX_MODE[4] = ~((MODE_[1] & (~MODE_[0])) | ((!MODE_[1]) & (!MODE_[0])));
	assign HEX_MODE[5] = ~((!MODE_[1]) & (!MODE_[0]));
	assign HEX_MODE[6] = ~((MODE_[1] & MODE_[0]) | (MODE_[1] & (~MODE_[0])));
	assign HEX_PFSM[0] = ~(((!PULSE_MACHINE[2]) & (!PULSE_MACHINE[1]) & (!PULSE_MACHINE[0])) |
								((!PULSE_MACHINE[2]) & PULSE_MACHINE[1] & (!PULSE_MACHINE[0])) |
								((!PULSE_MACHINE[2]) & PULSE_MACHINE[1] & PULSE_MACHINE[0]));
	assign HEX_PFSM[1] = ~(((!PULSE_MACHINE[2]) & (!PULSE_MACHINE[1]) & (!PULSE_MACHINE[0])) |
								((!PULSE_MACHINE[2]) & (!PULSE_MACHINE[1]) & PULSE_MACHINE[0]) |
								((!PULSE_MACHINE[2]) & PULSE_MACHINE[1] & (!PULSE_MACHINE[0])) |
								((!PULSE_MACHINE[2]) & PULSE_MACHINE[1] & PULSE_MACHINE[0]) |
								(PULSE_MACHINE[2] & (!PULSE_MACHINE[1]) & (!PULSE_MACHINE[0]))); 
	assign HEX_PFSM[2] = ~(((!PULSE_MACHINE[2]) & (!PULSE_MACHINE[1]) & (!PULSE_MACHINE[0])) |
								((!PULSE_MACHINE[2]) & (!PULSE_MACHINE[1]) & PULSE_MACHINE[0]) |
								((!PULSE_MACHINE[2]) & PULSE_MACHINE[1] & PULSE_MACHINE[0]) |
								(PULSE_MACHINE[2] & (!PULSE_MACHINE[1]) & (!PULSE_MACHINE[0])));
	assign HEX_PFSM[3] = ~(((!PULSE_MACHINE[2]) & (!PULSE_MACHINE[1]) & (!PULSE_MACHINE[0])) |
								((!PULSE_MACHINE[2]) & PULSE_MACHINE[1] & (!PULSE_MACHINE[0])) |
								((!PULSE_MACHINE[2]) & PULSE_MACHINE[1] & PULSE_MACHINE[0]));
	assign HEX_PFSM[4] = ~(((!PULSE_MACHINE[2]) & (!PULSE_MACHINE[1]) & (!PULSE_MACHINE[0])) |
								((!PULSE_MACHINE[2]) & PULSE_MACHINE[1] & (!PULSE_MACHINE[0])));
	assign HEX_PFSM[5] = ~(((!PULSE_MACHINE[2]) & (!PULSE_MACHINE[1]) & (!PULSE_MACHINE[0])) |
								(PULSE_MACHINE[2] & (!PULSE_MACHINE[1]) & (!PULSE_MACHINE[0])));
	assign HEX_PFSM[6] = ~(((!PULSE_MACHINE[2]) & PULSE_MACHINE[1] & (!PULSE_MACHINE[0])) |
								((!PULSE_MACHINE[2]) & PULSE_MACHINE[1] & PULSE_MACHINE[0]) |
								(PULSE_MACHINE[2] & (!PULSE_MACHINE[1]) & (!PULSE_MACHINE[0])));
	always@(posedge CLK or posedge RST) begin
		if (RST) begin
			WL_CTRL <= WL;
			BL_CTRL <= BL;
			COL_EN <= COL_EN_;
			ROW_EN <= ~ROW_EN_;
			WL_CTRL <= WL_SUB;
			BL_CTRL <= BL_SUB;
			COL_EN_SUB <= COL_EN_SUB_;
			ROW_EN_SUB <= ~ROW_EN_SUB_;
			DEBUG_COUNT <= START_POINT;
			DEBUG_OUT <= 0;
			BL_BUS_CHS <= 3'b000;
			WL_BUS_CHS[1] <= 1'b0;
			WL_BUS_CHS[0] <= 1'b0;
			SL_BUS_CHS <= 1'b0;
			BL_BUS_CHS_SUB <= 3'b000;
			WL_BUS_CHS_SUB[1] <= 1'b0;
			WL_BUS_CHS_SUB[0] <= 1'b0;
			SL_BUS_CHS_SUB <= 1'b0;
			COUNT <= 32'b0;
			PULSE_MACHINE <= 3'b000;
			FPGA_ENABLE <= 1'b0;
			Sense_Ctrl <= 1'b1;
			Reset <= 1'b1;
			CONN <= 1'b1;
			CAP_CTRL <= 1'b0;
			GEN_CTRL <= 1'b0;
			WHIGH_CTRL <= 1'b0;
			WLOW_CTRL <= 1'b1;
			FPGA_ENABLE_SUB <= 1'b0;
			Sense_Ctrl_SUB <= 1'b0;
			Reset_SUB <= 1'b0;
			CONN_SUB <= 1'b0;
			CAP_CTRL_SUB <= 1'b0;
			GEN_CTRL_SUB <= 1'b0;
			WHIGH_CTRL_SUB <= 1'b0;
			WLOW_CTRL_SUB <= 1'b0;
			COMMAND <= 2'b00;
			FIRST_COUNT <= 32'b0;
			Target <= 1'b0;
			Symbol <= 1'b0;
			CHECK_BYTE <= 8'b0;
			OUT_COUNT <= 32'b0011_0101_1001_1000_0110_1100_1011_1111;
			Para_Count <= 4'b1111;
		end
		else if (EN) begin
			WL_CTRL <= WL;
			BL_CTRL <= BL;
			COL_EN <= COL_EN_;
			WL_CTRL_SUB <= WL_SUB;
			BL_CTRL_SUB <= BL_SUB;
			COL_EN_SUB <= COL_EN_SUB_;
			Symbol <= Symbol_;
			if (Symbol != Target || DEBUG) begin
				if (MODE_ == 3'b000) begin // MANUAL SET RECOVER
					ROW_EN <= ~ROW_EN_;
					ROW_EN_SUB <= ~ROW_EN_SUB_;
					OUT_COUNT <= 32'b0;
					WL_BUS_CHS[1] <= 1'b0;
					WL_BUS_CHS[0] <= 1'b0;
					BL_BUS_CHS <= 3'b000;
					Para_Count <= 4'b0;
					case (PULSE_MACHINE)
						3'b000: begin
							SL_BUS_CHS <= 1'b0;
							if (COUNT > INITIAL) begin
								PULSE_MACHINE <= 3'b001;
								COUNT <= 32'b0;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
						3'b001: begin
							SL_BUS_CHS <= 1'b1;
							if (COUNT > SET_RECOVER_TIME) begin
								COUNT <= 32'b0;
								PULSE_MACHINE <= 3'b000;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
						default: begin
							SL_BUS_CHS <= 1'b0;
							COUNT <= 32'b0;
							Target <= ~Target;
	//						PULSE_MACHINE <= 3'b000;
						end
					endcase
				end
				else if (MODE_ == 3'b001) begin // MANUAL RESET RECOVER
					ROW_EN <= ~ROW_EN_;
					ROW_EN_SUB <= ~ROW_EN_SUB_;
					OUT_COUNT <= 32'b0;
					WL_BUS_CHS[1] <= 1'b0;
					WL_BUS_CHS[0] <= 1'b1;
					SL_BUS_CHS <= 1'b0;
					Para_Count <= 4'b0;
					case (PULSE_MACHINE)
						3'b000: begin
							BL_BUS_CHS <= 3'b000;
							if (COUNT > INITIAL) begin
								PULSE_MACHINE <= 3'b001;
								COUNT <= 32'b0;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
						3'b001: begin
							BL_BUS_CHS <= 3'b101;
							if (COUNT > RESET_RECOVER_TIME) begin
								COUNT <= 32'b0;
								PULSE_MACHINE <= 3'b111;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
						default: begin
							BL_BUS_CHS <= 3'b000;
							COUNT <= 32'b0;
							Target <= ~Target;
	//						PULSE_MACHINE <= 3'b000;
						end
					endcase
				end
				else if (MODE_ == 3'b010) begin // MANUAL WRITE PULSE GENERATION & Second Crossbar Write
					WL_BUS_CHS[1] <= 1'b0;
					WL_BUS_CHS[0] <= 1'b0;
					WL_BUS_CHS_SUB[1] <= 1'b0;
					WL_BUS_CHS_SUB[0] <= 1'b1;
					SL_BUS_CHS <= 1'b0;
					SL_BUS_CHS_SUB <= 1'b0;
					case (PULSE_MACHINE)
						3'b000: begin
							BL_BUS_CHS <= 3'b010;
							ROW_EN_SUB <= 4'b1111;
							ROW_EN <= ~ROW_EN_;
							FPGA_ENABLE <= 1'b1;
							Reset <= 1'b1;
							Sense_Ctrl <= 1'b1;
							CONN <= 1'b1;
							CAP_CTRL <= 1'b0;
							GEN_CTRL <= 1'b0;
							WHIGH_CTRL <= 1'b0;
							WLOW_CTRL <= 1'b1;
							OUT_COUNT <= 32'b0;
							BL_BUS_CHS_SUB <= 3'b000;
							if (COUNT > INITIAL) begin
								PULSE_MACHINE <= 3'b001;
								COUNT <= 32'b0;
								Para_Count <= 4'b0;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
						3'b001: begin
							BL_BUS_CHS <= 3'b010;
							FPGA_ENABLE <= 1'b1;
							ROW_EN_SUB <= 4'b1111;
							ROW_EN <= ~ROW_EN_;
							Reset <= 1'b0;
							Sense_Ctrl <= 1'b1;
							CONN <= 1'b1;
							CAP_CTRL <= 1'b0;
							GEN_CTRL <= 1'b0;
							WHIGH_CTRL <= 1'b0;
							WLOW_CTRL <= 1'b1;
							if (COUNT > DECLINE) begin
								PULSE_MACHINE <= 3'b010;
								COUNT <= 32'b0;
								Para_Count <= 4'b0;
							end
							else begin
								COUNT <= COUNT + 1;
							end
							if (OUT_COUNT > TEST_STEP) begin
								OUT_COUNT <= INT_BIAS;
								Para_Count <= Para_Count + 4'b0001;
							end
							else begin
								OUT_COUNT <= OUT_COUNT + 1;
							end
						end
						3'b010: begin
							BL_BUS_CHS <= 3'b010;
							BL_BUS_CHS_SUB <= 3'b001;
							ROW_EN <= 4'b1111;
							FPGA_ENABLE <= 1'b1;
							Reset <= 1'b0;
							Sense_Ctrl <= 1'b1;
							CONN <= 1'b1;
							CAP_CTRL <= 1'b0;
							GEN_CTRL <= 1'b0;
							WHIGH_CTRL <= 1'b1;
							WLOW_CTRL <= 1'b0;
//							Para_Count <= 4'b0;
							if (COUNT > RESET_STABLIZE_TIME) begin
								PULSE_MACHINE <= 3'b011;
								COUNT <= 32'b0;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
						3'b011: begin
							ROW_EN_SUB <= ~ROW_EN_SUB_;
							ROW_EN <= 4'b1111;
							if (COUNT > RESET_RECOVER_TIME) begin
								PULSE_MACHINE <= 3'b100;
								COUNT <= 32'b0;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
						default: begin
							BL_BUS_CHS <= 3'b000;
							BL_BUS_CHS_SUB <= 3'b000;
							FPGA_ENABLE <= 1'b0;
							Reset <= 1'b1;
							Sense_Ctrl <= 1'b1;
							CONN <= 1'b1;
							CAP_CTRL <= 1'b0;
							GEN_CTRL <= 1'b0;
							WHIGH_CTRL <= 1'b0;
							WLOW_CTRL <= 1'b1;
							if (COUNT > RESET_STABLIZE_TIME) begin
								Target <= ~Target;
//								PULSE_MACHINE <= 3'b000;
								COUNT <= 32'b0;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
					endcase
				end
				else if (MODE_ == 3'b011) begin // MANUAL BUFFER READ
					ROW_EN <= ~ROW_EN_;
					ROW_EN_SUB <= ~ROW_EN_SUB_;
					WL_BUS_CHS_SUB <= 2'b00;
					SL_BUS_CHS_SUB <= 1'b0;
					case (PULSE_MACHINE)
						3'b000: begin
							OUT_COUNT <= 32'b0;
							BL_BUS_CHS_SUB <= 3'b010;
							FPGA_ENABLE_SUB <= 1'b1;
							Reset_SUB <= 1'b1;
							Sense_Ctrl_SUB <= 1'b0;
							CONN_SUB <= 1'b0;
							CAP_CTRL_SUB <= 0;
							GEN_CTRL_SUB <= 1'b1;
							WHIGH_CTRL_SUB <= 1'b0;
							WLOW_CTRL_SUB <= 1'b1;
							if (COUNT > INITIAL) begin
								PULSE_MACHINE <= 3'b001;
								COUNT <= 32'b0;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
						3'b001: begin
							BL_BUS_CHS_SUB <= 3'b010;
							FPGA_ENABLE_SUB <= 1'b1;
							Reset_SUB <= 1'b0;
							Sense_Ctrl_SUB <= 1'b0;
							CONN_SUB <= 1'b0;
							CAP_CTRL_SUB <= 1'b1;
							GEN_CTRL_SUB <= 1'b0;
							WHIGH_CTRL_SUB <= 1'b0;
							WLOW_CTRL_SUB <= 1'b1;
							if (COUNT > READ_TIME) begin
								COUNT <= 32'b0;
								PULSE_MACHINE <= 3'b010;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
						3'b010: begin
							BL_BUS_CHS_SUB <= 3'b010;
							FPGA_ENABLE_SUB <= 1'b1;
							Reset_SUB <= 1'b0;
							Sense_Ctrl_SUB <= 1'b0;
							CONN_SUB <= 1'b0;
							CAP_CTRL_SUB <= 1'b0;
							GEN_CTRL_SUB <= 1'b0;
							WHIGH_CTRL_SUB <= 1'b0;
							WLOW_CTRL_SUB <= 1'b1;
							if (COUNT > NOLAP_TIME) begin
								COUNT <= 32'b0;
								PULSE_MACHINE <= 3'b011;
								DEBUG_COUNT <= START_POINT;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
						3'b011: begin
							BL_BUS_CHS_SUB <= 3'b010;
							FPGA_ENABLE_SUB <= 1'b1;
							Reset_SUB <= 1'b0;
							Sense_Ctrl_SUB <= 1'b0;
							CONN_SUB <= 1'b0;
							CAP_CTRL_SUB <= 1'b0;
							GEN_CTRL_SUB <= 1'b1;
							WHIGH_CTRL_SUB <= 1'b0;
							WLOW_CTRL_SUB <= 1'b1;
							if (COUNT > DISCHARGE_TIME) begin
								COUNT <= 32'b0;
								PULSE_MACHINE <= 3'b111;
							end
							else begin
								COUNT <= COUNT + 1;
							end
							if (DEBUG_COUNT >= SAMPLE_TIME) begin
								DEBUG_COUNT <= 1'b0;
								DEBUG_OUT <= DEBUG_IN;
								if (DEBUG_IN) begin
									OUT_COUNT <= OUT_COUNT + 1;
								end
							end
							else begin
								DEBUG_COUNT <= DEBUG_COUNT + 1;
							end
						end
						default: begin
							BL_BUS_CHS_SUB <= 3'b010;
							COUNT <= 32'b0;
							FPGA_ENABLE_SUB <= 1'b0;
							Reset_SUB <= 1'b0;
							Sense_Ctrl_SUB <= 1'b1;
							CONN_SUB <= 1'b0;
							CAP_CTRL_SUB <= 1'b0;
							GEN_CTRL_SUB <= 1'b0;
							WHIGH_CTRL_SUB <= 1'b0;
							WLOW_CTRL_SUB <= 1'b1;
							DEBUG_OUT <= 1'b0;
							Target <= ~Target;
							if (COUNT > RESET_STABLIZE_TIME) begin
								Target <= ~Target;
//								PULSE_MACHINE <= 3'b000;
								COUNT <= 32'b0;
							end
							else begin
								COUNT <= COUNT + 1;
							end
//							PULSE_MACHINE <= 3'b000;
						end
					endcase
				end
				else if (MODE_ == 3'b100) begin // MANUAL SET RECOVER SECOND
					ROW_EN <= ~ROW_EN_;
					ROW_EN_SUB <= ~ROW_EN_SUB_;
					OUT_COUNT <= 32'b0;
					WL_BUS_CHS_SUB[1] <= 1'b0;
					WL_BUS_CHS_SUB[0] <= 1'b0;
					BL_BUS_CHS_SUB <= 3'b000;
					case (PULSE_MACHINE)
						3'b000: begin
							FPGA_ENABLE <= 1'b1;
							SL_BUS_CHS_SUB <= 1'b0;
							if (COUNT > INITIAL) begin
								PULSE_MACHINE <= 3'b001;
								COUNT <= 32'b0;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
						3'b001: begin
							SL_BUS_CHS_SUB <= 1'b1;
							if (COUNT > SET_RECOVER_TIME) begin
								COUNT <= 32'b0;
								PULSE_MACHINE <= 3'b010;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
						default: begin
							SL_BUS_CHS_SUB <= 1'b0;
							FPGA_ENABLE <= 1'b0;
							COUNT <= 32'b0;
							Target <= ~Target;
							PULSE_MACHINE <= 3'b000;
						end
					endcase
				end
				else if (MODE_ == 3'b101) begin // MANUAL RESET RECOVER SECOND
					ROW_EN <= ~ROW_EN_;
					ROW_EN_SUB <= ~ROW_EN_SUB_;
					OUT_COUNT <= 32'b0;
					WL_BUS_CHS_SUB[1] <= 1'b0;
					WL_BUS_CHS_SUB[0] <= 1'b1;
					SL_BUS_CHS_SUB <= 1'b0;
					case (PULSE_MACHINE)
						3'b000: begin
							BL_BUS_CHS_SUB <= 3'b000;
							if (COUNT > INITIAL) begin
								PULSE_MACHINE <= 3'b001;
								COUNT <= 32'b0;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
						3'b001: begin
							BL_BUS_CHS_SUB <= 3'b101;
							if (COUNT > RESET_RECOVER_TIME) begin
								COUNT <= 32'b0;
								PULSE_MACHINE <= 3'b111;
							end
							else begin
								COUNT <= COUNT + 1;
							end
						end
						default: begin
							BL_BUS_CHS_SUB <= 3'b000;
							COUNT <= 32'b0;
							Target <= ~Target;
	//						PULSE_MACHINE <= 3'b000;
						end
					endcase
				end
//				else if (MODE_ == 3'b110) begin // Parallel Compute
//					case (PULSE_MACHINE)
//						3'b000: begin
//							WL_BUS_CHS <= 2'b00;
//							SL_BUS_CHS <= 1'b0;
//							BL_BUS_CHS <= 3'b010;
//							FPGA_ENABLE <= 1'b1;
//							Reset <= 1'b1;
//							Sense_Ctrl <= 1'b0;
//							CONN <= 1'b0;
//							CAP_CTRL <= 1'b0;
//							GEN_CTRL <= 1'b0;
//							WHIGH_CTRL <= 1'b0;
//							WLOW_CTRL <= 1'b1;
//							OUT_COUNT <= 32'b0;
//							if (COUNT > INITIAL) begin
//								PULSE_MACHINE <= 3'b001;
//								COUNT <= 32'b0;
//							end
//							else begin
//								COUNT <= COUNT + 1;
//							end
//							OUT_COUNT <= 32'b0;
//							Para_Count <= 4'b0;
//						end
//						3'b001: begin
//							Reset <= 1'b0;
//							Sense_Ctrl <= 1'b0;
//							CONN <= 1'b0;
//							CAP_CTRL <= 1'b1;
//							GEN_CTRL <= 1'b1;
//							WHIGH_CTRL <= 1'b0;
//							WLOW_CTRL <= 1'b1;
//							if (COUNT > PARA_COMPUTE_TIME) begin
//								COUNT <= 32'b0;
//								PULSE_MACHINE <= 3'b111;
//							end
//							else begin
//								COUNT <= COUNT + 1;
//							end
//							if (COMPUTE) begin
//								Para_Count <= Para_Count + 4'b0001;
//							end
//							if (COUNT_PORT) begin
//								OUT_COUNT <= OUT_COUNT + 1;
//							end
//						end
//						default: begin
//							COUNT <= 32'b0;
//							WL_BUS_CHS <= 2'b00;
//							SL_BUS_CHS <= 1'b0;
//							BL_BUS_CHS <= 3'b010;
//							FPGA_ENABLE <= 1'b0;
//							Reset <= 1'b1;
//							Sense_Ctrl <= 1'b0;
//							CONN <= 1'b0;
//							CAP_CTRL <= 1'b0;
//							GEN_CTRL <= 1'b0;
//							WHIGH_CTRL <= 1'b0;
//							WLOW_CTRL <= 1'b1;
//							Target <= ~Target;
////							PULSE_MACHINE <= 3'b000;
//						end
//					endcase
//				end
				else begin
					ROW_EN_SUB <= ~ROW_EN_SUB_;
					ROW_EN <= ~ROW_EN_;
//					ROW_EN <= 4'b0;
					Para_Count <= 4'b0;
					OUT_COUNT <= 0;
					if (WL_SELECT) begin
						OUT_COUNT <= 32'b0;
						WL_BUS_CHS[1] <= 1'b0;
						WL_BUS_CHS[0] <= 1'b1;
						BL_BUS_CHS <= 3'b000;
						case (PULSE_MACHINE)
							3'b000: begin
								SL_BUS_CHS <= 1'b0;
								if (COUNT > INITIAL) begin
									PULSE_MACHINE <= 3'b001;
									COUNT <= 32'b0;
								end
								else begin
									COUNT <= COUNT + 1;
								end
							end
							3'b001: begin
								SL_BUS_CHS <= 1'b1;
								if (COUNT > RESET_STABLIZE_TIME) begin
									COUNT <= 32'b0;
									PULSE_MACHINE <= 3'b010;
								end
								else begin
									COUNT <= COUNT + 1;
								end
							end
							default: begin
								SL_BUS_CHS <= 1'b0;
								COUNT <= 32'b0;
								Target <= ~Target;
		//						PULSE_MACHINE <= 3'b000;
							end
						endcase
					end
					else begin
						WL_BUS_CHS[1] <= 1'b0;
						WL_BUS_CHS[0] <= 1'b0;
						BL_BUS_CHS <= 3'b011;
						SL_BUS_CHS <= 1'b0;
						WL_BUS_CHS_SUB[1] <= 1'b0;
						WL_BUS_CHS_SUB[0] <= 1'b0;
						BL_BUS_CHS_SUB <= 3'b011;
						SL_BUS_CHS_SUB <= 1'b0;
						Target <= ~Target;
					end
				end
			end
			else begin
				ROW_EN_SUB <= ~ROW_EN_SUB_;
				CHECK_BYTE[0] <= Target;
				PULSE_MACHINE <= 3'b000;
			end
//			else begin
//				if (MODE == 3'b000) begin // Auto Write & Check Process
//					if (Symbol != Target) begin
//						if (COMMAND == 2'b00 || COMMAND == 2'b11) begin // READ_CHOICE
//							WL_BUS_CHS <= 2'b00;
//							SL_BUS_CHS <= 0;
//							BL_BUS_CHS <= 3'b010;
//							FPGA_ENABLE <= 1'b1;
//							Reset <= 1'b1;
//							Sense_Ctrl <= 1'b0;
//							CONN <= 1'b0;
//							CAP_CTRL <= 1'b1;
//							GEN_CTRL <= 1'b1;
//							WHIGH_CTRL <= 0;
//							WLOW_CTRL <= 1'b1;
//							case (PULSE_MACHINE)
//								3'b000: begin
//									OUT_COUNT <= 0;
//									if (COUNT > INITIAL) begin
//										PULSE_MACHINE <= 3'b001;
//										COUNT <= 0;
//									end
//									else begin
//										COUNT <= COUNT + 1;
//									end
//									OUT_COUNT <= 0;
//								end
//								3'b001: begin
//									Reset <= 1'b0;
//									Sense_Ctrl <= 1'b0;
//									CONN <= 1'b0;
//									CAP_CTRL <= 1'b1;
//									GEN_CTRL <= 1'b1;
//									WHIGH_CTRL <= 0;
//									WLOW_CTRL <= 1'b1;
//									if (COUNT > RESET_STABLIZE_TIME) begin
//										COUNT <= 0;
//										PULSE_MACHINE <= 3'b111;
//									end
//									else begin
//										COUNT <= COUNT + 1;
//									end
//									if (COUNT_PORT) begin
//										OUT_COUNT <= OUT_COUNT + 1;
//									end
//								end
//								default: begin
//									COUNT <= 0;
//									WL_BUS_CHS <= 2'b00;
//									SL_BUS_CHS <= 0;
//									BL_BUS_CHS <= 3'b010;
//									FPGA_ENABLE <= 1'b1;
//									Reset <= 1'b1;
//									Sense_Ctrl <= 1'b0;
//									CONN <= 1'b0;
//									CAP_CTRL <= 1'b0;
//									GEN_CTRL <= 1'b0;
//									WHIGH_CTRL <= 0;
//									WLOW_CTRL <= 1'b1;
//									if (COMMAND == 2'b00) begin
//										FIRST_COUNT <= OUT_COUNT;
//										if (OUT_COUNT > STANDARD && WRITING_VALUE[0] == 1) begin
//											COMMAND <= 2'b01;
//										end
//										else if (OUT_COUNT < STANDARD && WRITING_VALUE[0] == 0) begin
//											COMMAND <= 2'b10;
//										end
//										else begin
//											COMMAND <= 2'b11;
//										end
//									end
//									else begin
//										COMMAND <= 2'b00;
//										Target <= ~Target;
//									end
//									PULSE_MACHINE <= 0;
//								end
//							endcase
//						end
//						else if (COMMAND == 2'b01) begin // SET CHOICE
//							OUT_COUNT <= 0;
//							WL_BUS_CHS <= 2'b00;
//							BL_BUS_CHS <= 3'b000;
//							case (PULSE_MACHINE)
//								3'b000: begin
//									SL_BUS_CHS <= 0;
//									COUNT <= COUNT + 1;
//									if (COUNT > INITIAL) begin
//										PULSE_MACHINE <= 3'b001;
//										COUNT <= 0;
//									end
//								end
//								3'b001: begin
//									SL_BUS_CHS <= 1;
//									COUNT <= COUNT + 1;
//									if (COUNT > SET_RECOVER_TIME) begin
//										COUNT <= 0;
//										PULSE_MACHINE <= 3'b010;
//									end
//								end
//								default: begin
//									SL_BUS_CHS <= 0;
//									COUNT <= 0;
//									PULSE_MACHINE <= 3'b000;
//									COMMAND <= 2'b11;
//								end
//							endcase
//						end
//						else begin // RESET_CHOICE
//							OUT_COUNT <= 0;
//							WL_BUS_CHS <= 2'b01;
//							SL_BUS_CHS <= 0;
//							case (PULSE_MACHINE)
//								3'b000: begin
//									BL_BUS_CHS <= 3'b000;
//									COUNT <= COUNT + 1;
//									if (COUNT > INITIAL) begin
//										PULSE_MACHINE <= 3'b001;
//										COUNT <= 0;
//									end
//								end
//								3'b001: begin
//									BL_BUS_CHS <= 3'b101;
//									COUNT <= COUNT + 1;
//									if (COUNT > RESET_RECOVER_TIME) begin
//										COUNT <= 0;
//										PULSE_MACHINE <= 3'b111;
//									end
//								end
//								default: begin
//									BL_BUS_CHS <= 3'b000;
//									COUNT <= 0;
//									PULSE_MACHINE <= 3'b000;
//									COMMAND <= 2'b11;
//								end
//							endcase
//						end
//					end
//					else begin
//						CHECK_BYTE <= 8'hFF;
//					end
//				end
//				else if (MODE == 3'b001) begin // Test Available of First Crossbar
//					if (Symbol != Target) begin
//						if (COMMAND == 2'b00 || COMMAND == 2'b11) begin // READ_CHOICE
//							WL_BUS_CHS <= 2'b00;
//							SL_BUS_CHS <= 0;
//							BL_BUS_CHS <= 3'b010;
//							FPGA_ENABLE <= 1'b1;
//							Reset <= 1'b1;
//							Sense_Ctrl <= 1'b0;
//							CONN <= 1'b0;
//							CAP_CTRL <= 1'b1;
//							GEN_CTRL <= 1'b1;
//							WHIGH_CTRL <= 0;
//							WLOW_CTRL <= 1'b1;
//							case (PULSE_MACHINE)
//								3'b000: begin
//									OUT_COUNT <= 0;
//									if (COUNT > INITIAL) begin
//										PULSE_MACHINE <= 3'b001;
//										COUNT <= 0;
//									end
//									else begin
//										COUNT <= COUNT + 1;
//									end
//									OUT_COUNT <= 0;
//								end
//								3'b001: begin
//									Reset <= 1'b0;
//									Sense_Ctrl <= 1'b0;
//									CONN <= 1'b0;
//									CAP_CTRL <= 1'b1;
//									GEN_CTRL <= 1'b1;
//									WHIGH_CTRL <= 0;
//									WLOW_CTRL <= 1'b1;
//									if (COUNT > RESET_STABLIZE_TIME) begin
//										COUNT <= 0;
//										PULSE_MACHINE <= 3'b111;
//									end
//									else begin
//										COUNT <= COUNT + 1;
//									end
//									if (COUNT_PORT) begin
//										OUT_COUNT <= OUT_COUNT + 1;
//									end
//								end
//								default: begin
//									COUNT <= 0;
//									WL_BUS_CHS <= 2'b00;
//									SL_BUS_CHS <= 0;
//									BL_BUS_CHS <= 3'b010;
//									FPGA_ENABLE <= 1'b1;
//									Reset <= 1'b1;
//									Sense_Ctrl <= 1'b0;
//									CONN <= 1'b0;
//									CAP_CTRL <= 1'b0;
//									GEN_CTRL <= 1'b0;
//									WHIGH_CTRL <= 0;
//									WLOW_CTRL <= 1'b1;
//									if (COMMAND == 2'b00) begin
//										FIRST_COUNT <= OUT_COUNT;
//										if (OUT_COUNT > STANDARD) begin
//											COMMAND <= 2'b01;
//										end
//										else if (OUT_COUNT < STANDARD) begin
//											COMMAND <= 2'b10;
//										end
//										else begin
//											COMMAND <= 2'b11;
//										end
//									end
//									else begin
//										COMMAND <= 2'b00;
//										Target <= ~Target;
//									end
//									PULSE_MACHINE <= 0;
//								end
//							endcase
//						end
//						else if (COMMAND == 2'b01) begin // SET CHOICE
//							OUT_COUNT <= 0;
//							WL_BUS_CHS <= 2'b00;
//							BL_BUS_CHS <= 3'b000;
//							case (PULSE_MACHINE)
//								3'b000: begin
//									SL_BUS_CHS <= 0;
//									COUNT <= COUNT + 1;
//									if (COUNT > INITIAL) begin
//										PULSE_MACHINE <= 3'b001;
//										COUNT <= 0;
//									end
//								end
//								3'b001: begin
//									SL_BUS_CHS <= 1;
//									COUNT <= COUNT + 1;
//									if (COUNT > SET_RECOVER_TIME) begin
//										COUNT <= 0;
//										PULSE_MACHINE <= 3'b010;
//									end
//								end
//								default: begin
//									SL_BUS_CHS <= 0;
//									COUNT <= 0;
//									PULSE_MACHINE <= 3'b000;
//									COMMAND <= 2'b11;
//								end
//							endcase
//						end
//						else begin // RESET_CHOICE
//							OUT_COUNT <= 0;
//							WL_BUS_CHS <= 2'b01;
//							SL_BUS_CHS <= 0;
//							case (PULSE_MACHINE)
//								3'b000: begin
//									BL_BUS_CHS <= 3'b000;
//									COUNT <= COUNT + 1;
//									if (COUNT > INITIAL) begin
//										PULSE_MACHINE <= 3'b001;
//										COUNT <= 0;
//									end
//								end
//								3'b001: begin
//									BL_BUS_CHS <= 3'b101;
//									COUNT <= COUNT + 1;
//									if (COUNT > RESET_RECOVER_TIME) begin
//										COUNT <= 0;
//										PULSE_MACHINE <= 3'b111;
//									end
//								end
//								default: begin
//									BL_BUS_CHS <= 3'b000;
//									COUNT <= 0;
//									PULSE_MACHINE <= 3'b000;
//									COMMAND <= 2'b11;
//								end
//							endcase
//						end
//					end
//					else begin
//						CHECK_BYTE <= 8'hFF;
//					end
//				end
//				else if (MODE == 3'b010) begin // Test Available of Second Crossbar
//					if (Symbol_SUB != Target) begin
//						if (COMMAND == 2'b00 || COMMAND == 2'b11) begin // READ_CHOICE
//							WL_BUS_CHS_SUB <= 2'b00;
//							SL_BUS_CHS_SUB <= 0;
//							BL_BUS_CHS_SUB <= 3'b010;
//							FPGA_ENABLE_SUB <= 1'b1;
//							Reset_SUB <= 1'b1;
//							Sense_Ctrl_SUB <= 1'b0;
//							CONN_SUB <= 1'b0;
//							CAP_CTRL_SUB <= 1'b1;
//							GEN_CTRL_SUB <= 1'b1;
//							WHIGH_CTRL_SUB <= 0;
//							WLOW_CTRL_SUB <= 1'b1;
//							case (PULSE_MACHINE)
//								3'b000: begin
//									OUT_COUNT <= 0;
//									if (COUNT > INITIAL) begin
//										PULSE_MACHINE <= 3'b001;
//										COUNT <= 32'b0;
//									end
//									else begin
//										COUNT <= COUNT + 1;
//									end
//									OUT_COUNT <= 32'b0;
//								end
//								3'b001: begin
//									Reset_SUB <= 1'b0;
//									Sense_Ctrl_SUB <= 1'b0;
//									CONN_SUB <= 1'b0;
//									CAP_CTRL_SUB <= 1'b1;
//									GEN_CTRL_SUB <= 1'b1;
//									WHIGH_CTRL_SUB <= 0;
//									WLOW_CTRL_SUB <= 1'b1;
//									if (COUNT > RESET_STABLIZE_TIME) begin
//										COUNT <= 32'b0;
//										PULSE_MACHINE <= 3'b111;
//									end
//									else begin
//										COUNT <= COUNT + 1;
//									end
//									if (COUNT_PORT) begin
//										OUT_COUNT <= OUT_COUNT + 1;
//									end
//								end
//								default: begin
//									COUNT <= 0;
//									WL_BUS_CHS_SUB <= 2'b00;
//									SL_BUS_CHS_SUB <= 1'b0;
//									BL_BUS_CHS_SUB <= 3'b010;
//									FPGA_ENABLE_SUB <= 1'b1;
//									Reset_SUB <= 1'b1;
//									Sense_Ctrl_SUB <= 1'b0;
//									CONN_SUB <= 1'b0;
//									CAP_CTRL_SUB <= 1'b0;
//									GEN_CTRL_SUB <= 1'b0;
//									WHIGH_CTRL_SUB <= 0;
//									WLOW_CTRL_SUB <= 1'b1;
//									if (COMMAND == 2'b00) begin
//										FIRST_COUNT <= OUT_COUNT;
//										if (OUT_COUNT > STANDARD) begin
//											COMMAND <= 2'b01;
//										end
//										else if (OUT_COUNT < STANDARD) begin
//											COMMAND <= 2'b10;
//										end
//										else begin
//											COMMAND <= 2'b11;
//										end
//									end
//									else begin
//										COMMAND <= 2'b00;
//										Target <= ~Target;
//									end
//									PULSE_MACHINE <= 0;
//								end
//							endcase
//						end
//						else if (COMMAND == 2'b01) begin // SET CHOICE
//							OUT_COUNT <= 0;
//							WL_BUS_CHS_SUB <= 2'b00;
//							BL_BUS_CHS_SUB <= 3'b000;
//							case (PULSE_MACHINE)
//								3'b000: begin
//									SL_BUS_CHS_SUB <= 0;
//									COUNT <= COUNT + 1;
//									if (COUNT > INITIAL) begin
//										PULSE_MACHINE <= 3'b001;
//										COUNT <= 0;
//									end
//								end
//								3'b001: begin
//									SL_BUS_CHS_SUB <= 1'b1;
//									COUNT <= COUNT + 1;
//									if (COUNT > SET_RECOVER_TIME) begin
//										COUNT <= 0;
//										PULSE_MACHINE <= 3'b010;
//									end
//								end
//								default: begin
//									SL_BUS_CHS_SUB <= 1'b0;
//									COUNT <= 32'b0;
//									PULSE_MACHINE <= 3'b000;
//									COMMAND <= 2'b11;
//								end
//							endcase
//						end
//						else begin // RESET_CHOICE
//							OUT_COUNT <= 0;
//							WL_BUS_CHS_SUB <= 2'b01;
//							SL_BUS_CHS_SUB <= 0;
//							case (PULSE_MACHINE)
//								3'b000: begin
//									BL_BUS_CHS_SUB <= 3'b000;
//									COUNT <= COUNT + 1;
//									if (COUNT > INITIAL) begin
//										PULSE_MACHINE <= 3'b001;
//										COUNT <= 0;
//									end
//								end
//								3'b001: begin
//									BL_BUS_CHS_SUB <= 3'b101;
//									COUNT <= COUNT + 1;
//									if (COUNT > RESET_RECOVER_TIME) begin
//										COUNT <= 0;
//										PULSE_MACHINE <= 3'b111;
//									end
//								end
//								default: begin
//									BL_BUS_CHS_SUB <= 3'b000;
//									COUNT <= 0;
//									PULSE_MACHINE <= 3'b000;
//									COMMAND <= 2'b11;
//								end
//							endcase
//						end
//					end
//					else begin
//						CHECK_BYTE <= 8'hFF;
//					end
//				end
//				else begin// READ_ONLY
//					if (Symbol != Target) begin
//						case (PULSE_MACHINE)
//							3'b000: begin // Initialization of the 
//								WL_BUS_CHS <= 2'b00;
//								SL_BUS_CHS <= 1'b0;
//								BL_BUS_CHS <= 3'b010;
//								FPGA_ENABLE <= 1'b1;
//								Reset <= 1'b1;
//								Sense_Ctrl <= 1'b0;
//								CONN <= 1'b0;
//								CAP_CTRL <= 1'b1;
//								GEN_CTRL <= 1'b1;
//								WHIGH_CTRL <= 1'b0;
//								WLOW_CTRL <= 1'b1;
//								WL_BUS_CHS_SUB <= 2'b00;
//								SL_BUS_CHS_SUB <= 1'b0;
//								BL_BUS_CHS_SUB <= 3'b000;
//								FPGA_ENABLE_SUB <= 1'b0;
//								Reset_SUB <= 1'b0;
//								Sense_Ctrl_SUB <= 1'b0;
//								CONN_SUB <= 1'b0;
//								CAP_CTRL_SUB <= 1'b0;
//								GEN_CTRL_SUB <= 1'b0;
//								WHIGH_CTRL_SUB <= 1'b0;
//								WLOW_CTRL_SUB <= 1'b0;
//								OUT_COUNT <= 32'b0;
//								if (COUNT > INITIAL) begin
//									PULSE_MACHINE <= 3'b001;
//									COUNT <= 32'b0;
//								end
//								else begin
//									COUNT <= COUNT + 1;
//								end
//								OUT_COUNT <= 32'b0;
//								Para_Count <= 4'b0;
//							end
//							3'b001: begin // First Read Slope Generation, Second Set
//								BL_BUS_CHS <= 3'b010;
//								COUNT <= COUNT + 1;
//								FPGA_ENABLE <= 1'b1;
//								Reset <= 0;
//								Sense_Ctrl <= 1'b1;
//								CONN <= 1'b1;
//								CAP_CTRL <= 1'b0;
//								GEN_CTRL <= 1'b0;
//								WHIGH_CTRL <= 1'b0;
//								WLOW_CTRL <= 1'b1;
//								if (COUNT < SET_RECOVER_TIME) begin
//									SL_BUS_CHS_SUB <= 1'b1;
//								end
//								else begin
//									SL_BUS_CHS_SUB <= 1'b0;
//								end
//								if (COUNT > DECLINE) begin
//									PULSE_MACHINE <= 3'b010;
//									COUNT <= 32'b0;
//								end
//								if (OUT_COUNT > TEST_STEP) begin
//									OUT_COUNT <= 32'b0;
//									Para_Count <= Para_Count + 4'b0001;
//								end
//								else begin
//									OUT_COUNT <= OUT_COUNT + 1;
//								end
//							end
//							3'b010: begin // First Write Voltage Keep, Second Reset Initialize
//								BL_BUS_CHS <= 3'b000;
//								COUNT <= COUNT + 1;
//								FPGA_ENABLE <= 1'b1;
//								Reset <= 1'b0;
//								Sense_Ctrl <= 1'b0;
//								CONN <= 1'b0;
//								CAP_CTRL <= 1'b0;
//								GEN_CTRL <= 1'b0;
//								WHIGH_CTRL <= 1'b1;
//								WLOW_CTRL <= 1'b0;
//								OUT_COUNT <= 0;
//								WL_BUS_CHS_SUB <= 2'b00;
//								SL_BUS_CHS_SUB <= 0;
//								BL_BUS_CHS_SUB <= 3'b000;
//								COUNT <= COUNT + 1;
//								FPGA_ENABLE_SUB <= 1'b1;
//								Reset_SUB <= 1'b1;
//								Sense_Ctrl_SUB <= 1'b0;
//								CONN_SUB <= 1'b0;
//								CAP_CTRL_SUB <= 0;
//								GEN_CTRL_SUB <= 1'b1;
//								WHIGH_CTRL_SUB <= 0;
//								WLOW_CTRL_SUB <= 1'b1;
//								if (COUNT > INITIAL) begin
//									PULSE_MACHINE <= 3'b011;
//									COUNT <= 32'b0;
//								end
//							end
//							3'b011: begin // First Write Voltage Keep, Second Reset
//								BL_BUS_CHS <= 3'b000;
//								COUNT <= COUNT + 1;
//								FPGA_ENABLE <= 1'b1;
//								Reset <= 1'b0;
//								Sense_Ctrl <= 1'b0;
//								CONN <= 1'b0;
//								CAP_CTRL <= 1'b0;
//								GEN_CTRL <= 1'b0;
//								WHIGH_CTRL <= 1'b1;
//								WLOW_CTRL <= 1'b0;
//								OUT_COUNT <= 0;
//								WL_BUS_CHS_SUB <= 2'b00;
//								SL_BUS_CHS_SUB <= 0;
//								BL_BUS_CHS_SUB <= 3'b001;
//								COUNT <= COUNT + 1;
//								FPGA_ENABLE_SUB <= 1'b1;
//								Reset_SUB <= 1'b1;
//								Sense_Ctrl_SUB <= 1'b0;
//								CONN_SUB <= 1'b0;
//								CAP_CTRL_SUB <= 0;
//								GEN_CTRL_SUB <= 1'b1;
//								WHIGH_CTRL_SUB <= 0;
//								WLOW_CTRL_SUB <= 1'b1;
//								if (COUNT > RESET_RECOVER_TIME) begin
//									COUNT <= 32'b0;
//									PULSE_MACHINE <= 3'b100;
//								end
//							end
//							3'b100: begin // First Write Voltage Keep, Second Handle Buffer Read
//								BL_BUS_CHS <= 3'b000;
//								COUNT <= COUNT + 1;
//								FPGA_ENABLE <= 1'b1;
//								Reset <= 1'b0;
//								Sense_Ctrl <= 1'b0;
//								CONN <= 1'b0;
//								CAP_CTRL <= 1'b0;
//								GEN_CTRL <= 1'b0;
//								WHIGH_CTRL <= 1'b1;
//								WLOW_CTRL <= 1'b0;
//								BL_BUS_CHS_SUB <= 3'b010;
//								COUNT <= COUNT + 1;
//								FPGA_ENABLE_SUB <= 1'b1;
//								Reset_SUB <= 1'b0;
//								Sense_Ctrl_SUB <= 1'b0;
//								CONN_SUB <= 1'b0;
//								CAP_CTRL_SUB <= 1'b1;
//								GEN_CTRL_SUB <= 0;
//								WHIGH_CTRL_SUB <= 0;
//								WLOW_CTRL_SUB <= 1'b1;
//								if (COUNT > READ_TIME) begin
//									COUNT <= 32'b0;
//									PULSE_MACHINE <= 3'b101;
//								end
//							end
//							3'b101: begin // First Write Voltage Keep, Second Buffer Read
//								BL_BUS_CHS <= 3'b010;
//								COUNT <= COUNT + 1;
//								FPGA_ENABLE <= 1'b1;
//								Reset <= 1'b0;
//								Sense_Ctrl <= 1'b0;
//								CONN <= 1'b0;
//								CAP_CTRL <= 1'b0;
//								GEN_CTRL <= 1'b0;
//								WHIGH_CTRL <= 1'b1;
//								WLOW_CTRL <= 1'b0;
//								BL_BUS_CHS_SUB <= 3'b010;
//								COUNT <= COUNT + 1;
//								FPGA_ENABLE_SUB <= 1'b1;
//								Reset_SUB <= 1'b0;
//								Sense_Ctrl_SUB <= 1'b0;
//								CONN_SUB <= 1'b0;
//								CAP_CTRL_SUB <= 1'b0;
//								GEN_CTRL_SUB <= 0;
//								WHIGH_CTRL_SUB <= 0;
//								WLOW_CTRL_SUB <= 1'b1;
//								if (COUNT > NOLAP_TIME) begin
//									COUNT <= 0;
//									PULSE_MACHINE <= 3'b101;
//									DEBUG_COUNT <= START_POINT;
//								end
//							end
//							3'b110: begin // First Write Voltage Keep, Second Buffer Cap Discharge
//								BL_BUS_CHS <= 3'b010;
//								COUNT <= COUNT + 1;
//								FPGA_ENABLE <= 1'b1;
//								Reset <= 1'b0;
//								Sense_Ctrl <= 1'b0;
//								CONN <= 1'b0;
//								CAP_CTRL <= 1'b0;
//								GEN_CTRL <= 1'b0;
//								WHIGH_CTRL <= 1'b1;
//								WLOW_CTRL <= 1'b0;
//								BL_BUS_CHS_SUB <= 3'b010;
//								COUNT <= COUNT + 1;
//								FPGA_ENABLE_SUB <= 1'b1;
//								Reset_SUB <= 1'b0;
//								Sense_Ctrl_SUB <= 1'b0;
//								CONN_SUB <= 1'b0;
//								CAP_CTRL_SUB <= 1'b0;
//								GEN_CTRL_SUB <= 1'b1;
//								WHIGH_CTRL_SUB <= 1'b0;
//								WLOW_CTRL_SUB <= 1'b1;
//								if (COUNT > DISCHARGE_TIME) begin
//									COUNT <= 32'b0;
//									PULSE_MACHINE <= 3'b111;
//								end
//								if (DEBUG_COUNT >= SAMPLE_TIME) begin
//									DEBUG_COUNT <= 0;
//									DEBUG_OUT <= DEBUG_IN;
//									if (DEBUG_OUT) begin
//										OUT_COUNT <= OUT_COUNT + 1;
//									end
//								end
//								else begin
//									DEBUG_COUNT <= DEBUG_COUNT + 1;
//								end
//							end
//							default: begin
//								COUNT <= 32'b0;
//								WL_BUS_CHS <= 2'b00;
//								SL_BUS_CHS <= 1'b0;
//								BL_BUS_CHS <= 3'b000;
//								FPGA_ENABLE <= 1'b1;
//								Reset <= 1'b1;
//								Sense_Ctrl <= 1'b0;
//								CONN <= 1'b0;
//								CAP_CTRL <= 1'b0;
//								GEN_CTRL <= 1'b0;
//								WHIGH_CTRL <= 1'b0;
//								WLOW_CTRL <= 1'b1;
//								Target <= ~Target;
//								PULSE_MACHINE <= 3'b000;
//								BL_BUS_CHS_SUB <= 3'b000;
//								COUNT <= 0;
//								FPGA_ENABLE_SUB <= 1'b0;
//								Reset_SUB <= 0;
//								Sense_Ctrl_SUB <= 1'b1;
//								CONN_SUB <= 1'b0;
//								CAP_CTRL_SUB <= 0;
//								GEN_CTRL_SUB <= 0;
//								DEBUG_OUT <= 0;
//								WHIGH_CTRL_SUB <= 1'b0;
//								WLOW_CTRL_SUB <= 1'b1;
//							end
//						endcase
//					end
//					else begin
//						CHECK_BYTE <= 8'hFF;
//					end
//				end
//			end
		end
	end
endmodule
