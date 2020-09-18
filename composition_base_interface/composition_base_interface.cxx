#include <iostream>

template <typename T>
class ParentBase;

template <typename T>
class Parent;

template <typename T>
class Member;

template <typename T>
class ParentBase
{
public:
	using ValueType = T;

	ParentBase(){
		std::cout << "ParentBase address:   " << this << std::endl;
	}

	virtual ~ParentBase(){

	}

	virtual void print_value(){
		std::cout << "This is just the base class." << std::endl;
	}
};


template <typename T>
class Member
{
public:
	Member(ParentBase<T>& owner_base)
	: owner_base_ptr(&owner_base){
		std::cout << "Member address:   " << this << std::endl;
		std::cout << "ParentBase address in member:   " << owner_base_ptr << std::endl;
	}

	void do_something(){
		owner_base_ptr->print_value();
	}

private:
	ParentBase<T> * const owner_base_ptr;
};



template <typename T>
class Parent : public ParentBase<T>
{
public:
	using ValueType = T;

	Parent(T _value)
	: ParentBase<T>()
	  , member(*this)
	  , value (_value){
		std::cout << "Parent address:   " << this << std::endl;
	}

	void run(){
		member.do_something();
	}

	virtual void print_value() override {
			std::cout << "The value is:   " << value << std::endl;
		}
private:
	Member<T> member;
	ValueType value;
};

///////////////////////////////////

int main(int argc, char **argv) {

	Parent<double> parent_class(2.5);
	parent_class.run();

	return 0;
}
