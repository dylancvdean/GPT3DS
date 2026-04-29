#---------------------------------------------------------------------------------
.SUFFIXES:
#---------------------------------------------------------------------------------

ifeq ($(strip $(DEVKITPRO)),)
$(error "Please set DEVKITPRO in your environment. export DEVKITPRO=<path to>devkitPro")
endif

TOPDIR ?= $(CURDIR)
include $(DEVKITPRO)/devkitARM/3ds_rules

#---------------------------------------------------------------------------------
# TARGET is the name of the output
# BUILD is the directory where object files & intermediate files will be placed
# SOURCES is a list of directories containing source code
# DATA is a list of directories containing data files
# INCLUDES is a list of directories containing header files
# GRAPHICS is a list of directories containing graphics files
#   (assumes .t3s files that will be converted with .3ds Title Studio)
# ROMFS is the directory containing data to be added to RomFS, relative to the Makefile (Optional)
#
# NO_ICON: if set to anything, do not use icon.
# NO_NACP: if set to anything, no .nacp file is generated.
# APP_TITLE is the name of the app stored in the .nacp file (Optional)
# APP_AUTHOR is the author of the app stored in the .nacp file (Optional)
# APP_VERSION is the version of the app stored in the .nacp file (Optional)
# APP_TITLEID is the title ID of the app stored in the .nacp file (Optional)
# ICON is the file used for the icon, if NO_ICON is not defined
#---------------------------------------------------------------------------------
TARGET		:=	GPT3DS
BUILD		:=	build
SOURCES		:=	source
DATA		:=	data
INCLUDES	:=	source romfs
GRAPHICS	:=	
ROMFS		:=	romfs
MAKEROM		?=	$(firstword $(wildcard $(DEVKITPRO)/tools/bin/makerom tools/makerom /tmp/makerom-v0.18.4/makerom) makerom)
CHECKPOINT	?=	$(TOPDIR)/../ckpt_fineweb.pt
TOKENIZER_CBIN	?=	$(TOPDIR)/../tokenizer_qwends.cbin
PYTHON		?=	$(firstword $(wildcard $(TOPDIR)/../.venv/bin/python) python3)

APP_TITLE	:=	GPT3DS
APP_DESCRIPTION	:=	Tiny looped transformer
APP_AUTHOR	:=	GPT3DS Team
APP_VERSION	:=	0.1.0
APP_TITLEID	:=	0x000400000FF3FF00

#---------------------------------------------------------------------------------
# options for code generation
#---------------------------------------------------------------------------------
ARCH	:=	-march=armv6k -mtune=mpcore -mfloat-abi=hard -mtp=soft

CFLAGS	:=	-g -Wall -O3 -mword-relocations \
			-fomit-frame-pointer -ffunction-sections \
			$(ARCH)

CFLAGS	+=	$(INCLUDE) -D__3DS__

CXXFLAGS	:= $(CFLAGS) -fno-rtti -fno-exceptions -std=gnu++17

ASFLAGS	:=	-g $(ARCH)
LDFLAGS	=	-specs=3dsx.specs -g $(ARCH) -Wl,-Map,$(notdir $*.map)

LIBS	:= -lctru -lm

#---------------------------------------------------------------------------------
# list of directories containing libraries, this must be the top level containing
# include and lib
#---------------------------------------------------------------------------------
LIBDIRS	:= $(CTRULIB)

#---------------------------------------------------------------------------------
# no real need to edit anything past this point unless you need to add additional
# rules for different file extensions
#---------------------------------------------------------------------------------
ifneq ($(BUILD),$(notdir $(CURDIR)))
#---------------------------------------------------------------------------------

export OUTPUT	:=	$(CURDIR)/$(TARGET)
export TOPDIR	:=	$(CURDIR)

export VPATH	:=	$(foreach dir,$(SOURCES),$(CURDIR)/$(dir)) \
			$(foreach dir,$(GRAPHICS),$(CURDIR)/$(dir)) \
			$(foreach dir,$(DATA),$(CURDIR)/$(dir))

export DEPSDIR	:=	$(CURDIR)/$(BUILD)

CFILES		:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.c)))
CPPFILES	:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.cpp)))
SFILES		:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.s)))
PICAFILES	:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.v.pica)))
SHLISTFILES	:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.shlist)))
GFXFILES	:=	$(foreach dir,$(GRAPHICS),$(notdir $(wildcard $(dir)/*.t3s)))
BINFILES	:=	$(foreach dir,$(DATA),$(notdir $(wildcard $(dir)/*.*)))

#---------------------------------------------------------------------------------
# use CXX for linking C++ projects, CC for standard C
#---------------------------------------------------------------------------------
ifeq ($(strip $(CPPFILES)),)
#---------------------------------------------------------------------------------
	export LD	:=	$(CC)
#---------------------------------------------------------------------------------
else
#---------------------------------------------------------------------------------
	export LD	:=	$(CXX)
#---------------------------------------------------------------------------------
endif
#---------------------------------------------------------------------------------

export T3XFILES		:=	$(GFXFILES:.t3s=.t3x)

export OFILES_BIN	:=	$(addsuffix .o,$(BINFILES))
export OFILES_SRC	:=	$(CPPFILES:.cpp=.o) $(CFILES:.c=.o) $(SFILES:.s=.o)
export OFILES_DSP	:=	$(PICAFILES:.v.pica=.shbin.o) $(SHLISTFILES:.shlist=.shbin.o)
export OFILES		:=	$(OFILES_BIN) $(OFILES_SRC) $(OFILES_DSP)
export HFILES_BIN	:=	$(addsuffix .h,$(subst .,_,$(BINFILES)))

export INCLUDE	:=	$(foreach dir,$(INCLUDES),-I$(CURDIR)/$(dir)) \
			$(foreach dir,$(LIBDIRS),-I$(dir)/include) \
			-I$(CURDIR)/$(BUILD)

export LIBPATHS	:=	$(foreach dir,$(LIBDIRS),-L$(dir)/lib)
export _3DSXDEPS	:=	$(if $(NO_SMDH),,$(OUTPUT).smdh)

ifeq ($(strip $(ICON)),)
	export APP_ICON := $(CTRULIB)/default_icon.png
else
	export APP_ICON := $(TOPDIR)/$(ICON)
endif

ifeq ($(strip $(NO_SMDH)),)
	export _3DSXFLAGS += --smdh=$(CURDIR)/$(TARGET).smdh
endif

ifneq ($(ROMFS),)
	export _3DSXFLAGS += --romfs=$(CURDIR)/$(ROMFS)
endif

.PHONY: $(BUILD) clean all export-assets

#---------------------------------------------------------------------------------
all: export-assets $(BUILD)

export-assets: $(ROMFS)/model_weights.bin $(ROMFS)/model_desc.h $(ROMFS)/tokenizer_qwends.cbin

$(ROMFS)/model_weights.bin $(ROMFS)/model_desc.h &: export_3ds.py $(CHECKPOINT)
	@echo export model from $(notdir $(CHECKPOINT)) ...
	@$(PYTHON) export_3ds.py --checkpoint $(CHECKPOINT) --out_dir $(ROMFS)

$(ROMFS)/tokenizer_qwends.cbin: $(TOKENIZER_CBIN)
	@echo update $(notdir $@) ...
	@cp $< $@

$(BUILD):
	@[ -d $@ ] || mkdir -p $@
	@$(MAKE) --no-print-directory -C $(BUILD) -f $(CURDIR)/Makefile

#---------------------------------------------------------------------------------
clean:
	@echo clean ...
	@rm -fr $(BUILD) $(TARGET).3dsx $(OUTPUT).smdh $(TARGET).elf $(TARGET).cia

#---------------------------------------------------------------------------------
else

DEPENDS	:=	$(OFILES:.o=.d)

#---------------------------------------------------------------------------------
# main targets
#---------------------------------------------------------------------------------
all	:	$(OUTPUT).3dsx $(OUTPUT).cia

$(OUTPUT).3dsx	:	$(OUTPUT).elf $(_3DSXDEPS)

$(OUTPUT).cia	:	$(OUTPUT).elf $(OUTPUT).smdh $(TOPDIR)/app.rsf $(shell find $(TOPDIR)/$(ROMFS) -type f)
	@command -v $(MAKEROM) >/dev/null 2>&1 || { echo "makerom not found. Install MakeROM or run make MAKEROM=/path/to/makerom"; exit 1; }
	$(SILENTCMD)$(MAKEROM) -f cia -target t -o $@ -elf $(OUTPUT).elf -icon $(OUTPUT).smdh -rsf $(TOPDIR)/app.rsf -desc app:4 -ver 0 -DROMFS_ROOT=$(TOPDIR)/$(ROMFS)
	$(SILENTMSG) built ... $(notdir $@)

$(OUTPUT).elf	:	$(OFILES)

$(OFILES_SRC)	:	$(HFILES_BIN)

#---------------------------------------------------------------------------------
# you need a rule like this for each extension you use as binary data
#---------------------------------------------------------------------------------
%.bin.o	%_bin.h :	%.bin
#---------------------------------------------------------------------------------
	@echo $(notdir $<)
	@$(bin2o)

#---------------------------------------------------------------------------------
%.t3x.o	%_t3x.h :	%.t3x
#---------------------------------------------------------------------------------
	@echo $(notdir $<)
	@$(bin2o)

#---------------------------------------------------------------------------------
%.shbin.o %_shbin.h :	%.shbin
#---------------------------------------------------------------------------------
	@echo $(notdir $<)
	@$(bin2o)

#---------------------------------------------------------------------------------
%.shbin	:	%.v.pica %.shlist
#---------------------------------------------------------------------------------
	@echo $(notdir $<)
	$(eval CURBIN := $(patsubst %.v.pica,%.shbin,$(notdir $<)))
	$(eval CURH := $(patsubst %.v.pica,%_shbin.h,$(notdir $<)))
	@picasso $< -o $(CURBIN)
	@$(bin2o)

#---------------------------------------------------------------------------------
%.t3x	:	%.t3s
#---------------------------------------------------------------------------------
	@echo $(notdir $<)
	@tex3ds -i $< -H $(patsubst %.t3s,%_t3x.h,$(notdir $<)) -d $(DEPSDIR)/$*.d -o $@

-include $(DEPENDS)

#---------------------------------------------------------------------------------------
endif
#---------------------------------------------------------------------------------------
